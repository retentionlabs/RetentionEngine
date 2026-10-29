"""Restoration training: distill a full teacher into a layer-trimmed student."""

import logging
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

logger = logging.getLogger(__name__)


def _import_transformer_engine():
    try:
        import transformer_engine.pytorch as te
    except ImportError as e:
        raise ImportError(
            "The FP8 teacher path needs NVIDIA Transformer Engine: "
            "`pip install 'retentionengine[fp8]'` (CUDA GPU required)."
        ) from e
    return te


def convert_to_fp8_layers(
    module: nn.Module, skip_modules: Sequence[str] = ("lm_head",)
) -> nn.Module:
    """Replace ``nn.Linear`` layers with Transformer Engine ``te.Linear`` in place, keeping weights.

    Only GEMMs run in FP8, so norms (RMSNorm in Llama/Qwen) are left untouched.
    Skipped: modules whose name is in ``skip_modules`` (``lm_head`` by default),
    linears whose weight is tied to the input embeddings, and linears whose
    dimensions are not multiples of 16 (an FP8 GEMM requirement).
    Run the converted model under ``fp8_autocast`` (DistillationTrainer does
    this when ``use_fp8_teacher=True``).
    """
    te = _import_transformer_engine()

    tied_ptrs = set()
    get_emb = getattr(module, "get_input_embeddings", None)
    if callable(get_emb) and get_emb() is not None:
        tied_ptrs.add(get_emb().weight.data_ptr())

    converted = 0

    def _convert(parent: nn.Module) -> None:
        nonlocal converted
        for name, child in parent.named_children():
            if name in skip_modules:
                continue
            if not isinstance(child, nn.Linear):
                _convert(child)
                continue
            if child.weight.data_ptr() in tied_ptrs:
                continue
            if child.in_features % 16 or child.out_features % 16:
                continue
            device = (
                child.weight.device if child.weight.is_cuda else torch.device("cuda")
            )
            new = te.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                params_dtype=child.weight.dtype,
                device=device,
            )
            with torch.no_grad():
                new.weight.copy_(child.weight)
                if child.bias is not None:
                    new.bias.copy_(child.bias)
            setattr(parent, name, new)
            converted += 1

    _convert(module)
    logger.info("Converted %d nn.Linear layers to te.Linear for FP8.", converted)
    return module


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Token-level KD loss: ``T^2 * mean_t KL(p_teacher || p_student)`` at temperature ``T``.

    KL is summed over the vocabulary and averaged over valid tokens. With
    ``labels``, logits at position ``t`` predict token ``t+1`` (causal LM), so the
    valid positions are ``labels[:, 1:] != -100`` on ``logits[:, :-1]`` -- the same
    tokens the LM loss covers. Without labels, ``attention_mask`` (or every
    position) defines the valid tokens.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Student/teacher logits shape mismatch: {tuple(student_logits.shape)} vs "
            f"{tuple(teacher_logits.shape)} (vocabularies must match)."
        )
    teacher_logits = teacher_logits.to(student_logits.device)
    if labels is not None:
        student_logits = student_logits[:, :-1]
        teacher_logits = teacher_logits[:, :-1]
        mask = labels[:, 1:] != -100
    elif attention_mask is not None:
        mask = attention_mask.bool()
    else:
        mask = torch.ones(
            student_logits.shape[:-1], dtype=torch.bool, device=student_logits.device
        )
    mask = mask.to(student_logits.device)

    student = student_logits[mask].float()
    teacher = teacher_logits[mask].float()
    if student.shape[0] == 0:
        return student_logits.sum() * 0.0

    log_p_student = F.log_softmax(student / temperature, dim=-1)
    log_p_teacher = F.log_softmax(teacher / temperature, dim=-1)
    kl = F.kl_div(log_p_student, log_p_teacher, reduction="none", log_target=True).sum(
        dim=-1
    )
    return kl.mean() * temperature**2


class DistillationTrainer(Trainer):
    """Trainer for restoration KD: ``loss = (1 - alpha) * LM + alpha * T^2 * KL``.

    The teacher runs in the dtype it was loaded with (bf16 recommended). FP8 is
    opt-in via ``use_fp8_teacher=True`` (needs Transformer Engine and a GPU with
    FP8 support). ``labels`` are never passed to the teacher.
    """

    # compute_loss returns a per-micro-batch mean; the Trainer divides it by the
    # gradient accumulation steps.
    loss_is_scaled_for_ga = False

    def __init__(
        self,
        *args,
        teacher_model: nn.Module,
        alpha: float = 1.0,
        temperature: float = 2.0,
        use_fp8_teacher: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        # Same GA scaling on transformers versions without `loss_is_scaled_for_ga`,
        # and keeps the Trainer from counting `num_items_in_batch` we do not use.
        self.model_accepts_loss_kwargs = False

        self.alpha = alpha
        self.temperature = temperature
        self.use_fp8_teacher = use_fp8_teacher

        if use_fp8_teacher:
            teacher_model = convert_to_fp8_layers(teacher_model)
        if getattr(teacher_model, "hf_device_map", None) is None:
            teacher_model.to(self.args.device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model = teacher_model

    def _teacher_logits(self, inputs) -> torch.Tensor:
        teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        with torch.no_grad():
            if self.use_fp8_teacher:
                te = _import_transformer_engine()
                fp8_autocast = getattr(te, "fp8_autocast", None) or te.autocast
                with fp8_autocast(enabled=True):
                    return self.teacher_model(**teacher_inputs, use_cache=False).logits
            return self.teacher_model(**teacher_inputs, use_cache=False).logits

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs
    ):
        student_outputs = model(**inputs)
        loss_kd = distillation_loss(
            student_outputs.logits,
            self._teacher_logits(inputs),
            labels=inputs.get("labels"),
            attention_mask=inputs.get("attention_mask"),
            temperature=self.temperature,
        )
        loss_lm = student_outputs.loss
        if loss_lm is None:
            if self.alpha < 1.0:
                raise ValueError(
                    "alpha < 1 needs `labels` in the batch for the LM loss."
                )
            loss = self.alpha * loss_kd
        else:
            loss = (1.0 - self.alpha) * loss_lm + self.alpha * loss_kd
        return (loss, student_outputs) if return_outputs else loss
