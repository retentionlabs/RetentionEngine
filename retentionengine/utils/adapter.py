import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 loss_fn: nn.Module = nn.CrossEntropyLoss(), device: str = "cuda"):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.model.to(self.device)

    def train_step(self, batch: tuple) -> float:
        self.model.train()
        inputs, targets = batch

        inputs = inputs[:, :-1].to(self.device)
        targets = targets[:, 1:].to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)

        loss = self.loss_fn(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_step(self, batch: tuple) -> float:
        self.model.eval()
        with torch.no_grad():
            inputs, targets = batch

            inputs = inputs[:, :-1].to(self.device)
            targets = targets[:, 1:].to(self.device)

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

        return loss.item()

    def train_epoch(self, dataloader: DataLoader) -> float:
        total_loss = 0
        for batch in dataloader:
            loss = self.train_step(batch)
            total_loss += loss
        return total_loss / len(dataloader)

    def eval_epoch(self, dataloader: DataLoader) -> float:
        total_loss = 0
        for batch in dataloader:
            loss = self.eval_step(batch)
            total_loss += loss
        return total_loss / len(dataloader)