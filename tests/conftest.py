import os

# CPU-only smoke tests: hide GPUs before torch is imported and never touch the Hub.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("WANDB_DISABLED", "true")

import pytest  # noqa: E402
import torch  # noqa: E402
from transformers import (  # noqa: E402
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

VOCAB_SIZE = 128


def _tiny_llama() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return LlamaForCausalLM(config).eval()


def _tiny_qwen3() -> Qwen3ForCausalLM:
    # Sliding window on the last two layers so `layer_types` is non-trivial.
    config = Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        use_sliding_window=True,
        sliding_window=16,
        max_window_layers=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return Qwen3ForCausalLM(config).eval()


MODEL_FACTORIES = {"llama": _tiny_llama, "qwen3": _tiny_qwen3}


@pytest.fixture(params=sorted(MODEL_FACTORIES))
def tiny_model(request):
    torch.manual_seed(0)
    return MODEL_FACTORIES[request.param]()


@pytest.fixture
def tiny_llama():
    torch.manual_seed(0)
    return _tiny_llama()
