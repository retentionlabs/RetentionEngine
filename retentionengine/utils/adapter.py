import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
import wandb
import os
import transformer_engine.pytorch as te


def convert_to_fp8_layers(module: nn.Module) -> nn.Module:
    """
    Replaces nn.Linear and nn.LayerNorm with Transformer Engine's FP8-supported layers.
    """
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
            use_wandb: bool = True,
            checkpoint_dir: str = "./checkpoints",
    ):
        self.adapt_model = adapt_model
        
        # Convert the teacher model's layers to FP8-supported layers.
        print("INFO: Converting base_model to support FP8 with Transformer Engine...")
        self.base_model = convert_to_fp8_layers(base_model)
        print("INFO: Conversion complete.")
        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.distillation_loss_fn = distillation_loss_fn
        self.lm_loss_fn = lm_loss_fn
        self.temperature = temperature
        self.alpha = alpha
        self.use_wandb = use_wandb
        
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_eval_loss = float('inf')
        self.train_dataloader_len = 0
        
    def save_checkpoint(self, epoch: int, eval_loss: float = None):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}_checkpoint.pth")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.adapt_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'eval_loss': eval_loss,
            'best_eval_loss': self.best_eval_loss
        }, checkpoint_path)
        print(f"INFO: Checkpoint saved to {checkpoint_path}")

    def save_best_model(self, eval_loss: float, epoch: int):
        if eval_loss < self.best_eval_loss:
            print(f"INFO: New best model found! Eval Loss: {eval_loss:.4f} (Previous: {self.best_eval_loss:.4f})")
            self.best_eval_loss = eval_loss
            best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.adapt_model.state_dict(),
                'eval_loss': eval_loss
            }, best_model_path)
            print(f"INFO: Best model saved to {best_model_path}")

    def train_step(self, batch: dict) -> tuple:
        primary_device = next(self.adapt_model.parameters()).device
        batch = {k: v.to(primary_device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        self.adapt_model.train()
        self.optimizer.zero_grad()
        
        # Apply te.fp8_autocast only to the teacher model.
        with torch.no_grad():
            with te.fp8_autocast(enabled=True):
                base_outputs = self.base_model(**batch)
                base_logits = base_outputs.logits
        
        # The student model operates in its original precision (e.g., bfloat16).
        adapt_outputs = self.adapt_model(**batch)
        adapt_logits = adapt_outputs.logits

        # Loss calculation
        soft_targets = F.softmax(base_logits / self.temperature, dim=-1)
        soft_prob = F.log_softmax(adapt_logits / self.temperature, dim=-1)
        distillation_loss = self.distillation_loss_fn(soft_prob, soft_targets) * (self.temperature ** 2)
        
        lm_loss = self.lm_loss_fn(
            adapt_logits.view(-1, adapt_logits.size(-1)),
            batch['labels'].view(-1)
        )
        
        total_loss = self.alpha * distillation_loss + (1. - self.alpha) * lm_loss

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
                
                # Also apply te.fp8_autocast only to the teacher model during evaluation.
                with te.fp8_autocast(enabled=True):
                    base_outputs = self.base_model(**batch)
                    base_logits = base_outputs.logits

                adapt_outputs = self.adapt_model(**batch)
                adapt_logits = adapt_outputs.logits

                soft_targets = F.softmax(base_logits / self.temperature, dim=-1)
                soft_prob = F.log_softmax(adapt_logits / self.temperature, dim=-1)
                distillation_loss = self.distillation_loss_fn(soft_prob, soft_targets) * (self.temperature ** 2)
                
                lm_loss = self.lm_loss_fn(
                    adapt_logits.view(-1, adapt_logits.size(-1)),
                    batch['labels'].view(-1)
                )
                
                total_loss = self.alpha * distillation_loss + (1. - self.alpha) * lm_loss
                
                total_eval_loss += total_loss.item()
        
        avg_eval_loss = total_eval_loss / len(dataloader)
        print(f"Epoch [{epoch}/{num_epochs}] Eval Avg Loss: {avg_eval_loss:.4f}")

        if self.use_wandb:
            global_step = epoch * self.train_dataloader_len
            wandb.log({"eval/epoch_loss": avg_eval_loss}, step=global_step)

        return avg_eval_loss
            
    def run_training(self, train_dataloader: DataLoader, num_epochs: int, eval_dataloader: DataLoader = None):
        self.train_dataloader_len = len(train_dataloader)
        for epoch in range(1, num_epochs + 1):
            self.train_epoch(train_dataloader, epoch, num_epochs)
            if eval_dataloader:
                avg_eval_loss = self.evaluate_epoch(eval_dataloader, epoch, num_epochs)
                self.save_best_model(avg_eval_loss, epoch)
                self.save_checkpoint(epoch, avg_eval_loss)
            else:
                self.save_checkpoint(epoch)