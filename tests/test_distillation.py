import copy
import math
import sys
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from conftest import VOCAB_SIZE
from torch.utils.data import Dataset
from transformers import TrainingArguments

from retentionengine import RetentionEngine, RetentionEngineConfig
from retentionengine.utils import DistillationTrainer, convert_to_fp8_layers
from retentionengine.utils.adapter import distillation_loss


class _ToyLMDataset(Dataset):
    def __init__(self, n=8, length=12, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.items = []
        for i in range(n):
            ids = torch.randint(3, VOCAB_SIZE, (length,), generator=g)
            mask = torch.ones(length, dtype=torch.long)
            pad = i % 3  # right padding on some rows
            if pad:
                ids[-pad:] = 0
                mask[-pad:] = 0
            labels = ids.clone()
            labels[mask == 0] = -100
            self.items.append(
                {"input_ids": ids, "attention_mask": mask, "labels": labels}
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class _NoLabelsTeacher(nn.Module):
    """Wraps a teacher and fails if it ever receives `labels`."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        assert "labels" not in kwargs
        return self.model(**kwargs)


def _args(tmp_path, **overrides):
    kwargs = dict(
        output_dir=str(tmp_path),
        max_steps=2,
        per_device_train_batch_size=2,
        learning_rate=1e-3,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        seed=0,
    )
    kwargs.update(overrides)
    return TrainingArguments(**kwargs)


def _losses(trainer):
    return [log["loss"] for log in trainer.state.log_history if "loss" in log]


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_distillation_trainer_two_steps(tiny_llama, tmp_path, alpha):
    teacher = copy.deepcopy(tiny_llama)
    student = RetentionEngine(
        tiny_llama, RetentionEngineConfig(layers_to_remove=1)
    ).model
    before = [p.detach().clone() for p in student.parameters()]

    trainer = DistillationTrainer(
        model=student,
        teacher_model=_NoLabelsTeacher(teacher),
        args=_args(tmp_path),
        train_dataset=_ToyLMDataset(),
        alpha=alpha,
        temperature=2.0,
    )
    result = trainer.train()

    assert trainer.state.global_step == 2
    losses = _losses(trainer)
    assert len(losses) == 2 and all(math.isfinite(x) for x in losses)
    assert math.isfinite(result.training_loss)
    assert any(not torch.equal(b, p) for b, p in zip(before, student.parameters()))
    assert all(not p.requires_grad for p in teacher.parameters())


def test_distillation_trainer_with_engine_and_grad_accum(tiny_llama, tmp_path):
    teacher = copy.deepcopy(tiny_llama)
    engine = RetentionEngine(tiny_llama, RetentionEngineConfig(layers_to_remove=1))
    trainer = DistillationTrainer(
        model=engine,
        teacher_model=teacher,
        args=_args(
            tmp_path, gradient_accumulation_steps=2, remove_unused_columns=False
        ),
        train_dataset=_ToyLMDataset(),
    )
    trainer.train()
    losses = _losses(trainer)
    assert len(losses) == 2 and all(math.isfinite(x) for x in losses)


def test_distillation_loss_masks_and_scales():
    torch.manual_seed(0)
    student = torch.randn(2, 5, VOCAB_SIZE)
    teacher = torch.randn(2, 5, VOCAB_SIZE)
    labels = torch.randint(0, VOCAB_SIZE, (2, 5))
    labels[1, 3:] = -100

    assert distillation_loss(teacher, teacher, labels).item() == pytest.approx(
        0.0, abs=1e-6
    )

    # Reference: shift (logits[t] predicts labels[t+1]), keep labels != -100, mean over tokens.
    T = 2.0
    s, t, keep = student[:, :-1], teacher[:, :-1], labels[:, 1:] != -100
    kl = (
        F.softmax(t / T, -1) * (F.log_softmax(t / T, -1) - F.log_softmax(s / T, -1))
    ).sum(-1)
    expected = kl[keep].mean() * T**2
    assert distillation_loss(
        student, teacher, labels, temperature=T
    ).item() == pytest.approx(expected.item(), rel=1e-5)

    # Masked positions do not change the loss.
    perturbed = student.clone()
    perturbed[1, 2:] += 10.0
    assert distillation_loss(
        perturbed, teacher, labels, temperature=T
    ).item() == pytest.approx(expected.item(), rel=1e-5)


def test_fp8_conversion_requires_transformer_engine(tiny_llama, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformer_engine", None)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", None)
    with pytest.raises(ImportError, match="fp8"):
        convert_to_fp8_layers(tiny_llama)


def test_fp8_refuses_device_mapped_teacher(tiny_llama, tmp_path, monkeypatch):
    # Refused before Transformer Engine is imported or any layer is replaced.
    monkeypatch.setitem(sys.modules, "transformer_engine", None)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", None)
    teacher = copy.deepcopy(tiny_llama)
    teacher.hf_device_map = {"": "cpu"}
    student = RetentionEngine(
        tiny_llama, RetentionEngineConfig(layers_to_remove=1)
    ).model

    with pytest.raises(ValueError, match="device_map"):
        DistillationTrainer(
            model=student,
            teacher_model=teacher,
            args=_args(tmp_path),
            train_dataset=_ToyLMDataset(),
            use_fp8_teacher=True,
        )
    with pytest.raises(ValueError, match="device_map"):
        convert_to_fp8_layers(_NoLabelsTeacher(teacher))  # map on a submodule
    assert all(
        type(m) is nn.Linear for m in teacher.modules() if isinstance(m, nn.Linear)
    )


def test_fp8_conversion_keeps_weights(monkeypatch):
    class FakeTELinear(nn.Linear):
        def __init__(
            self, in_features, out_features, bias=True, params_dtype=None, device=None
        ):
            super().__init__(in_features, out_features, bias=bias, dtype=params_dtype)

    te_pkg = types.ModuleType("transformer_engine")
    te_pt = types.ModuleType("transformer_engine.pytorch")
    te_pt.Linear = FakeTELinear
    te_pkg.pytorch = te_pt
    monkeypatch.setitem(sys.modules, "transformer_engine", te_pkg)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te_pt)

    from conftest import _tiny_llama

    torch.manual_seed(0)
    model = _tiny_llama()
    model.config.tie_word_embeddings = True
    model.lm_head.weight = model.model.embed_tokens.weight
    input_ids = torch.randint(3, VOCAB_SIZE, (1, 6))
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits
    q_proj = model.model.layers[0].self_attn.q_proj.weight.detach().clone()

    convert_to_fp8_layers(model)  # the fake ignores `device`, so this stays on CPU

    assert isinstance(model.model.layers[0].self_attn.q_proj, FakeTELinear)
    assert isinstance(model.model.layers[0].mlp.down_proj, FakeTELinear)
    assert type(model.lm_head) is nn.Linear  # skipped: lm_head / tied to embeddings
    assert torch.equal(model.model.layers[0].self_attn.q_proj.weight, q_proj)
    with torch.no_grad():
        assert torch.allclose(model(input_ids=input_ids).logits, expected, atol=1e-6)
