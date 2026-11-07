import json
import os
import sys

import pytest
import torch
from conftest import VOCAB_SIZE
from transformers import AutoModelForCausalLM

import retentionengine
from retentionengine import (
    ModelSizeManager,
    RetentionEngine,
    RetentionEngineConfig,
    layer_block_indices,
    remove_layers,
)
from retentionengine.adapters.engine import ENGINE_CONFIG_NAME


def _input_ids(batch=2, length=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, VOCAB_SIZE, (batch, length), generator=g)


def _layer_indices(model):
    return [layer.self_attn.layer_idx for layer in model.model.layers]


def test_import_does_not_need_thelethe():
    assert "thelethe" not in sys.modules
    assert hasattr(retentionengine, "RetentionEngine")


def test_layer_block_indices_default_is_deep_block():
    # Keeps the first layers and the final layer; removes the block right before the last one.
    assert layer_block_indices(8, 3) == [4, 5, 6]
    assert layer_block_indices(8, 2, start=0) == [0, 1]
    assert layer_block_indices(8, 0) == []
    with pytest.raises(ValueError):
        layer_block_indices(4, 4)
    with pytest.raises(ValueError):
        layer_block_indices(8, 3, start=6)


def test_trim_forward_and_cached_generation(tiny_model):
    original_layers = list(tiny_model.model.layers)
    engine = RetentionEngine(tiny_model, RetentionEngineConfig(layers_to_remove=2))

    # Default position: layers 1 and 2 removed, final layer (3) kept.
    assert engine.config.removed_layer_indices == [1, 2]
    assert [id(m) for m in engine.model.model.layers] == [
        id(original_layers[0]),
        id(original_layers[3]),
    ]
    assert engine.model.config.num_hidden_layers == 2
    assert _layer_indices(engine.model) == [0, 1]
    layer_types = getattr(engine.model.config, "layer_types", None)
    if layer_types is not None:
        assert len(layer_types) == 2

    input_ids = _input_ids()
    with torch.no_grad():
        out = engine(input_ids=input_ids, labels=input_ids)
    assert out.logits.shape == (2, 8, VOCAB_SIZE)
    assert torch.isfinite(out.loss)

    gen_kwargs = dict(max_new_tokens=6, do_sample=False, pad_token_id=0)
    with torch.no_grad():
        cached = engine.generate(input_ids, use_cache=True, **gen_kwargs)
        uncached = engine.generate(input_ids, use_cache=False, **gen_kwargs)
    assert cached.shape == (2, 14)
    assert torch.equal(cached, uncached)


@pytest.mark.parametrize("removal_start", [None, 0])
def test_qwen3_trimmed_cached_generation_past_sliding_window(removal_start):
    from conftest import _tiny_qwen3

    torch.manual_seed(0)
    model = _tiny_qwen3()
    window = model.config.sliding_window
    engine = RetentionEngine(
        model, RetentionEngineConfig(layers_to_remove=1, removal_start=removal_start)
    )
    sliding = [
        layer.self_attn.sliding_window is not None for layer in model.model.layers
    ]
    assert any(sliding) and not all(sliding)  # hybrid full/sliding stack

    # 8 prompt + 32 new tokens = 40, well past the 16-token window.
    input_ids = _input_ids()
    gen_kwargs = dict(
        max_new_tokens=2 * window,
        do_sample=False,
        pad_token_id=0,
        return_dict_in_generate=True,
        output_logits=True,
    )
    with torch.no_grad():
        cached = engine.generate(input_ids, use_cache=True, **gen_kwargs)
        uncached = engine.generate(input_ids, use_cache=False, **gen_kwargs)

    total = input_ids.shape[1] + 2 * window
    assert cached.sequences.shape == (2, total)
    assert torch.equal(cached.sequences, uncached.sequences)
    # Logits too: the window changes them once the sequence exceeds it, so a cache
    # that mixed up the (renumbered) full/sliding layers would not match.
    assert torch.allclose(
        torch.stack(cached.logits), torch.stack(uncached.logits), atol=1e-5
    )

    cache_layers = getattr(cached.past_key_values, "layers", None)
    if cache_layers is not None:  # transformers versions with per-layer caches
        kv_lens = [layer.keys.shape[-2] for layer in cache_layers]
        assert len(kv_lens) == len(sliding)
        for kv_len, is_sliding in zip(kv_lens, sliding):
            assert kv_len <= window if is_sliding else kv_len == total - 1


@pytest.mark.parametrize("removal_start, kept", [(None, [0, 1, 3]), (0, [1, 2, 3])])
def test_qwen3_sliding_window_layout_survives_trim_and_reload(
    tmp_path, removal_start, kept
):
    from conftest import _tiny_qwen3

    model = _tiny_qwen3()
    windows = [layer.self_attn.sliding_window for layer in model.model.layers]
    assert windows == [None, None, 16, 16]
    expected = [windows[i] for i in kept]

    engine = RetentionEngine(
        model, RetentionEngineConfig(layers_to_remove=1, removal_start=removal_start)
    )
    layer_types = getattr(model.config, "layer_types", None)
    if layer_types is not None:  # transformers versions with per-layer types
        assert layer_types == [
            "sliding_attention" if w else "full_attention" for w in expected
        ]
    assert model.config.max_window_layers == expected.count(None)

    # Reloading rebuilds attention modules from the config: the layout must not shift.
    engine.save_pretrained(str(tmp_path))
    reloaded = AutoModelForCausalLM.from_pretrained(str(tmp_path))
    assert [
        layer.self_attn.sliding_window for layer in reloaded.model.layers
    ] == expected


def test_front_removal_is_configurable(tiny_llama):
    engine = RetentionEngine(
        tiny_llama, RetentionEngineConfig(layers_to_remove=1, removal_start=0)
    )
    assert engine.config.removed_layer_indices == [0]
    assert _layer_indices(engine.model) == [0, 1, 2]


def test_remove_layers_arbitrary_indices(tiny_llama):
    remove_layers(tiny_llama, [0, 2])
    assert tiny_llama.config.num_hidden_layers == 2
    assert _layer_indices(tiny_llama) == [0, 1]


def test_no_trimming_by_default(tiny_llama):
    engine = RetentionEngine(tiny_llama)
    assert engine.model.config.num_hidden_layers == 4
    assert engine.config.removed_layer_indices == []


def test_auto_resize_uses_budget(tiny_llama):
    total = ModelSizeManager.count_parameters(tiny_llama)
    per_layer = ModelSizeManager.params_per_layer(tiny_llama)
    assert per_layer == sum(p.numel() for p in tiny_llama.model.layers[0].parameters())

    config = RetentionEngineConfig(max_total_params=total - per_layer, min_layers=2)
    engine = RetentionEngine(tiny_llama, config, auto_resize=True)
    assert engine.config.layers_to_remove == 1
    assert ModelSizeManager.count_parameters(engine) <= total - per_layer

    with pytest.raises(ValueError):
        RetentionEngine(
            tiny_llama, RetentionEngineConfig(layers_to_remove=1), auto_resize=True
        )


def test_min_layers_must_keep_a_layer(tiny_llama):
    # min_layers=0 would let auto_resize remove every layer.
    with pytest.raises(ValueError, match="min_layers"):
        RetentionEngineConfig(max_total_params=1, min_layers=0)
    with pytest.raises(ValueError, match="min_layers"):
        ModelSizeManager.calculate_layers_to_remove(tiny_llama, 0, 1, min_layers=0)

    # min_layers=1 under an impossible budget keeps exactly one layer.
    engine = RetentionEngine(
        tiny_llama,
        RetentionEngineConfig(max_total_params=1, min_layers=1),
        auto_resize=True,
    )
    assert engine.model.config.num_hidden_layers == 1


def test_size_info_counts_each_param_once(tiny_llama):
    engine = RetentionEngine(tiny_llama, RetentionEngineConfig(layers_to_remove=1))
    info = engine.get_model_size_info()
    expected = sum(
        p.numel() for p in {id(p): p for p in tiny_llama.parameters()}.values()
    )
    assert info["base_model_params"] == expected
    assert info["total_params"] == expected
    assert info["module_params"] == 0
    # Registered once: no duplicate state_dict keys from a second reference to the base model.
    assert all(k.startswith("model.") for k in engine.state_dict())


def test_memory_module_is_stubbed(tiny_llama):
    engine = RetentionEngine(tiny_llama)
    with pytest.raises(NotImplementedError):
        engine.attach_memory_module()


def test_save_load_roundtrip(tiny_model, tmp_path):
    engine = RetentionEngine(tiny_model, RetentionEngineConfig(layers_to_remove=1))
    engine.save_pretrained(str(tmp_path))

    with open(tmp_path / "config.json") as f:
        hf_config = json.load(f)
    assert hf_config["num_hidden_layers"] == 3
    assert not any(k.startswith("size_") for k in hf_config)
    assert os.path.isfile(tmp_path / ENGINE_CONFIG_NAME)

    input_ids = _input_ids()
    with torch.no_grad():
        expected = engine(input_ids=input_ids).logits

    loaded = RetentionEngine.from_pretrained(str(tmp_path))
    assert loaded.config.removed_layer_indices == engine.config.removed_layer_indices
    assert loaded.config.original_num_layers == 4
    assert loaded.model.config.num_hidden_layers == 3
    assert _layer_indices(loaded.model) == [0, 1, 2]
    with torch.no_grad():
        assert torch.allclose(loaded(input_ids=input_ids).logits, expected, atol=1e-5)

    # The checkpoint is a plain HF model as well.
    plain = AutoModelForCausalLM.from_pretrained(str(tmp_path)).eval()
    with torch.no_grad():
        assert torch.allclose(plain(input_ids=input_ids).logits, expected, atol=1e-5)
        out = plain.generate(
            input_ids, max_new_tokens=4, do_sample=False, pad_token_id=0
        )
    assert out.shape == (2, 12)


def test_trimmed_record_rejects_untrimmed_model(tiny_llama, tmp_path):
    engine = RetentionEngine(tiny_llama, RetentionEngineConfig(layers_to_remove=1))
    from conftest import _tiny_llama

    with pytest.raises(ValueError):
        RetentionEngine(_tiny_llama(), engine.config)
