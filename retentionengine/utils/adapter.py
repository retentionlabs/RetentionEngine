# 파일명: huggingface_adapter.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
import transformer_engine.pytorch as te

def convert_to_fp8_layers(module: nn.Module) -> nn.Module:
    """nn.Linear와 nn.LayerNorm을 Transformer Engine의 FP8 지원 레이어로 변환합니다."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, te.Linear(
                child.in_features,
                child.out_features,
                bias=(child.bias is not None),
            ))
        elif isinstance(child, nn.LayerNorm):
            setattr(module, name, te.LayerNorm(
                child.normalized_shape,
                eps=child.eps,
            ))
        else:
            convert_to_fp8_layers(child)
    return module

class DistillationTrainer(Trainer):
    """
    KL Divergence Loss를 사용한 지식 증류(복원 학습)를 위한 커스텀 Trainer.
    """
    def __init__(self, *args, teacher_model, alpha=1.0, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        
        print("INFO: Teacher 모델을 FP8 지원 레이어로 변환합니다...")
        self.teacher_model = convert_to_fp8_layers(teacher_model)
        print("INFO: 변환 완료.")

        self.alpha = alpha
        self.temperature = temperature
        
        self.teacher_model.to(self.model.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        손실 계산 로직: 학생 모델의 LM Loss와 선생님-학생 간 Distillation Loss를 조합.
        """
        # 학생(Student) 모델의 출력 및 기본 손실(LM Loss) 계산
        student_outputs = model(**inputs)
        loss_lm = student_outputs.loss
        logits_student = student_outputs.logits

        # 선생님(Teacher) 모델의 출력을 FP8로 빠르게 계산
        with torch.no_grad():
            with te.fp8_autocast(enabled=True):
                teacher_outputs = self.teacher_model(**inputs)
                logits_teacher = teacher_outputs.logits

        # KL Divergence Loss를 사용한 Distillation Loss 계산
        distillation_loss_fn = nn.KLDivLoss(reduction="batchmean")
        
        soft_targets = F.softmax(logits_teacher / self.temperature, dim=-1)
        soft_prob = F.log_softmax(logits_student / self.temperature, dim=-1)
        
        loss_distill = distillation_loss_fn(soft_prob, soft_targets) * (self.temperature ** 2)

        # 최종 손실: alpha=1.0으로 설정 시, 오직 두 모델을 똑같이 만드는 데 집중
        loss = (1. - self.alpha) * loss_lm + self.alpha * loss_distill

        return (loss, student_outputs) if return_outputs else loss