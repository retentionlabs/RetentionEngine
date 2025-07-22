import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm


class Adapter:
    def __init__(
            self,
            adapt_model: nn.Module,
            base_model: nn.Module,
            optimizer: optim.Optimizer,
            distillation_loss_fn: nn.Module = nn.KLDivLoss(reduction='batchmean'),
            # distillation loss with base model
            temperature: float = 1.0,
            device: str = "cuda"
    ):
        self.adapt_model = adapt_model.to(device)
        self.base_model = base_model.to(device)
        self.optimizer = optimizer
        self.distillation_loss_fn = distillation_loss_fn
        self.temperature = temperature
        self.device = device

        # Freeze base model parameters
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

    def train_step(self, batch: tuple) -> tuple[float, torch.Tensor, torch.Tensor]:
        """
        Perform a single training step on a batch.

        Returns:
            loss: batch loss value
            predictions: model predictions
            targets: actual targets (for accuracy calculation)
        """
        self.adapt_model.train()
        inputs, targets = batch

        # Process inputs and targets once
        inputs = inputs[:, :-1].to(self.device)
        targets = targets[:, 1:].to(self.device)

        # Get base model outputs (frozen)
        with torch.no_grad():
            base_outputs = self.base_model(inputs)

        # Get adapt model outputs
        adapt_outputs = self.adapt_model(inputs)

        # Calculate distillation loss
        loss_soft = self.distillation_loss_fn(
            F.log_softmax(adapt_outputs / self.temperature, dim=-1).reshape(-1, adapt_outputs.size(-1)),
            F.softmax(base_outputs / self.temperature, dim=-1).reshape(-1, base_outputs.size(-1))
        )
        total_loss = loss_soft * (self.temperature ** 2)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Calculate predictions (detach to save memory)
        with torch.no_grad():
            predictions = adapt_outputs.argmax(dim=-1).detach()

        return total_loss.item(), predictions, targets

    def eval_step(self, batch: tuple) -> tuple[float, torch.Tensor, torch.Tensor]:
        """
        Perform a single evaluation step on a batch.

        Returns:
            loss: batch loss value
            predictions: model predictions
            targets: actual targets
        """
        self.adapt_model.eval()

        with torch.no_grad():
            inputs, targets = batch
            inputs = inputs[:, :-1].to(self.device)
            targets = targets[:, 1:].to(self.device)

            # Forward pass for both models
            adapt_outputs = self.adapt_model(inputs)
            base_outputs = self.base_model(inputs)

            # Calculate loss
            loss = self.distillation_loss_fn(
                F.log_softmax(adapt_outputs / self.temperature, dim=-1).reshape(-1, adapt_outputs.size(-1)),
                F.softmax(base_outputs / self.temperature, dim=-1).reshape(-1, base_outputs.size(-1))
            )
            loss = loss * (self.temperature ** 2)

            predictions = adapt_outputs.argmax(dim=-1)

        return loss.item(), predictions, targets

    def train_epoch(self, dataloader: DataLoader, epoch: int, num_epochs: int) -> float:
        """
        Train for one complete epoch.

        Args:
            dataloader: training data loader
            epoch: current epoch number
            num_epochs: total number of epochs

        Returns:
            average loss for the epoch
        """
        total_loss = 0
        total_steps = 0
        correct = 0
        total = 0

        with tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch") as pbar:
            for batch in pbar:
                # Single step call to get all information
                loss, predictions, targets = self.train_step(batch)

                total_loss += loss
                total_steps += 1

                # Calculate accuracy without redundant computation
                correct += predictions.eq(targets).sum().item()
                total += targets.numel()

                # Update progress bar with real-time metrics
                pbar.set_postfix({
                    'epoch': epoch,
                    'loss': f'{loss:.4f}',
                    'acc': f'{(correct / total) * 100:.2f}%'
                })

        return total_loss / total_steps

    def eval_epoch(self, dataloader: DataLoader, epoch: int, num_epochs: int) -> float:
        """
        Evaluate for one complete epoch.

        Args:
            dataloader: evaluation data loader
            epoch: current epoch number
            num_epochs: total number of epochs

        Returns:
            average loss for the epoch
        """
        total_loss = 0
        total_steps = 0
        correct = 0
        total = 0

        with tqdm(dataloader, desc=f"Eval {epoch}/{num_epochs}", unit="batch") as pbar:
            for batch in pbar:
                # Single step call to get all information
                loss, predictions, targets = self.eval_step(batch)

                total_loss += loss
                total_steps += 1

                # Calculate accuracy without redundant computation
                correct += predictions.eq(targets).sum().item()
                total += targets.numel()

                # Update progress bar with real-time metrics
                pbar.set_postfix({
                    'epoch': epoch,
                    'loss': f'{loss:.4f}',
                    'acc': f'{(correct / total) * 100:.2f}%'
                })

        return total_loss / total_steps