from thelethe.titans import PretrainedTitansConfig, PreTrainedTitansModel
from transformers import PreTrainedModel, AutoTokenizer
from torch.optim import AdamW
from torch import nn

from ..utils import Trainer
from ..data import get_oasst_dataloader


class RetentionEngine(nn.Module):
    def __init__(self, basemodel: PreTrainedModel, config: PretrainedTitansConfig):
        super().__init__()
        self.module = PreTrainedTitansModel(config)
        self.module.model = basemodel

        # Combine the configurations
        self.config = config
        if hasattr(basemodel.config, 'to_dict'):
            config_dict = basemodel.config.to_dict()
            for key, value in config_dict.items():
                self.config.__dict__[key] = value
        elif hasattr(basemodel.config, '__dict__'):
            for key, value in basemodel.config.__dict__.items():
                if not key.startswith('_'):  # exclude private attributes
                    self.config.__dict__[key] = value

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(
        self,
        tokenizer: AutoTokenizer,
        epochs: int = 1,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        train_lang: str = None,
        eval_lang: str = None,
        device: str = "cuda"
    ):

        train_dataloader = get_oasst_dataloader(tokenizer, batch_size, split='train', lang=train_lang)
        eval_dataloader = None
        if eval_lang:
            eval_dataloader = get_oasst_dataloader(tokenizer, batch_size, split='validation', lang=eval_lang)

        optimizer = AdamW(self.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        trainer = Trainer(
            model=self,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device
        )

        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

            avg_train_loss = trainer.train_epoch(train_dataloader)
            print(f"Average Training Loss: {avg_train_loss:.4f}")

            if eval_dataloader:
                avg_eval_loss = trainer.eval_epoch(eval_dataloader)
                print(f"Average Validation Loss: {avg_eval_loss:.4f}")


    @classmethod
    def from_pretrained(
            cls,
            model_id: str,
            basemodel: PreTrainedModel
    ) -> "RetentionEngine":
        """
        Load a RetentionEngine from a pretrained model and configuration.
        """
        config = PretrainedTitansConfig.from_pretrained(model_id)
        module = PreTrainedTitansModel.from_pretrained(model_id)
        module.model = basemodel
        engine = cls(basemodel, config)
        engine.module = module
        return engine

    def save_pretrained(self, save_directory: str):
        """
        Save the RetentionEngine to a directory.
        """
        self.module.save_pretrained(save_directory)
        self.config.save_pretrained(save_directory)
