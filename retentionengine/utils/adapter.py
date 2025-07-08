import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm 


class Adapter:
    def __init__(self,
                 adapt_model: nn.Module,
                 base_model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 hard_loss_fn: nn.Module = nn.CrossEntropyLoss(),#실제답과의 loss
                 distillation_loss_fn: nn.Module = nn.KLDivLoss(reduction='batchmean'), #base model과의 distillation loss
                 temperature: float = 2.0,
                 alpha: float = 0.5, #hard loss와 soft loss의 가중치
                 device: str = "cuda"):

        self.adapt_model = adapt_model.to(device)
        self.base_model = base_model.to(device)
        self.optimizer = optimizer
        self.hard_loss_fn = hard_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.temperature = temperature
        self.alpha = alpha
        self.device = device

        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False # base model의 파라미터는 학습하지 않음

    def train_step(self, batch: tuple) -> float:
        self.adapt_model.train()
        inputs, targets = batch

        inputs = inputs[:, :-1].to(self.device)
        targets = targets[:, 1:].to(self.device)

        with torch.no_grad():
            base_outputs = self.base_model(inputs)

        adapt_outputs = self.adapt_model(inputs)

        loss_hard = self.hard_loss_fn(
            adapt_outputs.reshape(-1, adapt_outputs.size(-1)),
            targets.reshape(-1)
        )

        loss_soft = self.distillation_loss_fn(
            F.log_softmax(adapt_outputs / self.temperature, dim=-1).reshape(-1, adapt_outputs.size(-1)),
            F.softmax(base_outputs / self.temperature, dim=-1).reshape(-1, base_outputs.size(-1))
        )
        loss_soft = loss_soft * (self.temperature ** 2)

        total_loss = self.alpha * loss_hard + (1 - self.alpha) * loss_soft

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def eval_step(self, batch: tuple) -> float:
        self.adapt_model.eval()
        with torch.no_grad():
            inputs, targets = batch
            inputs = inputs[:, :-1].to(self.device)
            targets = targets[:, 1:].to(self.device)

            outputs = self.adapt_model(inputs)
            loss = self.hard_loss_fn(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

        return loss.item()

    def train_epoch(self, dataloader: DataLoader) -> float:
        total_loss = 0

        for batch in tqdm(dataloader, desc="Training"):
            loss = self.train_step(batch)
            total_loss += loss
        return total_loss / len(dataloader)

    def eval_epoch(self, dataloader: DataLoader) -> float:
        total_loss = 0

        for batch in tqdm(dataloader, desc="Evaluating"):
            loss = self.eval_step(batch)
            total_loss += loss
        return total_loss / len(dataloader)