# adapter.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
import wandb

# FP8 학습에 권장되는 데이터 타입 (E4M3: 정밀도가 더 높아 학습 안정성에 유리)
FP8_DTYPE = torch.float8_e4m3fn 

class Adapter:
    def __init__(
            self,
            adapt_model: nn.Module,
            base_model: nn.Module,
            optimizer: optim.Optimizer,
            lr_scheduler,
            distillation_loss_fn: nn.Module,
            lm_loss_fn: nn.Module,
            temperature: float = 2.0,
            alpha: float = 0.5,
            use_wandb: bool = True
    ):
        self.adapt_model = adapt_model
        self.base_model = base_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.distillation_loss_fn = distillation_loss_fn
        self.lm_loss_fn = lm_loss_fn
        self.temperature = temperature
        self.alpha = alpha
        self.use_wandb = use_wandb
        self.train_dataloader_len = 0

        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

    def train_step(self, batch: dict) -> tuple:
        primary_device = next(self.adapt_model.parameters()).device
        batch = {k: v.to(primary_device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        self.adapt_model.train()

        with torch.autocast(device_type='cuda', dtype=FP8_DTYPE):
            with torch.no_grad():
                base_outputs = self.base_model(**batch)
                base_logits = base_outputs.logits
            
            adapt_outputs = self.adapt_model(**batch)
            adapt_logits = adapt_outputs.logits
            
            soft_targets = F.softmax(base_logits / self.temperature, dim=-1)
            soft_prob = F.log_softmax(adapt_logits / self.temperature, dim=-1)
            distillation_loss = self.distillation_loss_fn(soft_prob, soft_targets) * (self.temperature ** 2)
            
            lm_loss = self.lm_loss_fn(adapt_logits.view(-1, adapt_logits.size(-1)), batch['labels'].view(-1))
            
            total_loss = self.alpha * distillation_loss + (1. - self.alpha) * lm_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        
        return total_loss.item(), distillation_loss.item(), lm_loss.item()
    
    def train_epoch(self, dataloader: DataLoader, epoch: int, num_epochs: int):
        total_epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs} [T]", unit="batch")
        
        for step, batch in enumerate(progress_bar):
            loss, dist_loss, lm_loss = self.train_step(batch)
            total_epoch_loss += loss
            
            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'dist_loss': f'{dist_loss:.4f}',
                'lm_loss': f'{lm_loss:.4f}'
            })
            
            if self.use_wandb:
                global_step = (epoch - 1) * len(dataloader) + step
                wandb.log({
                    "train/step_loss": loss,
                    "train/distillation_loss": dist_loss,
                    "train/lm_loss": lm_loss,
                    "train/learning_rate": self.lr_scheduler.get_last_lr()[0]
                }, step=global_step)

        avg_epoch_loss = total_epoch_loss / len(dataloader)
        print(f"Epoch [{epoch}/{num_epochs}] Train Avg Loss: {avg_epoch_loss:.4f}")
        
        if self.use_wandb:
            global_step = epoch * len(dataloader)
            wandb.log({"train/epoch_loss": avg_epoch_loss}, step=global_step)

    def evaluate_epoch(self, dataloader: DataLoader, epoch: int, num_epochs: int):
        print("INFO: Starting evaluation...")
        self.adapt_model.eval()
        total_eval_loss = 0
        primary_device = next(self.adapt_model.parameters()).device
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs} [E]", unit="batch")
        
        with torch.no_grad():
            for batch in progress_bar:
                batch = {k: v.to(primary_device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                with torch.autocast(device_type='cuda', dtype=FP8_DTYPE):
                    base_outputs = self.base_model(**batch)
                    base_logits = base_outputs.logits

                    adapt_outputs = self.adapt_model(**batch)
                    adapt_logits = adapt_outputs.logits

                    soft_targets = F.softmax(base_logits / self.temperature, dim=-1)
                    soft_prob = F.log_softmax(adapt_logits / self.temperature, dim=-1)
                    distillation_loss = self.distillation_loss_fn(soft_prob, soft_targets) * (self.temperature ** 2)
                    
                    lm_loss = self.lm_loss_fn(adapt_logits.view(-1, adapt_logits.size(-1)), batch['labels'].view(-1))
                    total_loss = self.alpha * distillation_loss + (1. - self.alpha) * lm_loss
                
                total_eval_loss += total_loss.item()
        
        avg_eval_loss = total_eval_loss / len(dataloader)
        print(f"Epoch [{epoch}/{num_epochs}] Eval Avg Loss: {avg_eval_loss:.4f}")

        if self.use_wandb:
            global_step = epoch * self.train_dataloader_len
            wandb.log({"eval/epoch_loss": avg_eval_loss}, step=global_step)
            
    def run_training(self, train_dataloader: DataLoader, num_epochs: int, eval_dataloader: DataLoader = None):
        self.train_dataloader_len = len(train_dataloader)
        for epoch in range(1, num_epochs + 1):
            self.train_epoch(train_dataloader, epoch, num_epochs)
            if eval_dataloader:
                self.evaluate_epoch(eval_dataloader, epoch, num_epochs)