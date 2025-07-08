from thelethe.titans import PretrainedTitansConfig, PreTrainedTitansModel
from transformers import PreTrainedModel, AutoTokenizer
from torch import nn
from torch.optim import AdamW

from ..utils import Adapter
from ..datasets import get_pg19_dataloader


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

    def adapt(self,
        tokenizer: AutoTokenizer,
        epochs: int = 3,
        batch_size: int = 1,
        max_length: int = 8192,
        learning_rate: float = 2e-5,

        alpha: float = 0.3,
        temperature: float = 2.0
    ):

        device = self.module.device
        # 2. 모델 역할 정의 및 설정 🧑‍🏫
        adapt_model = self.module
        base_model = self.module.model
        base_model.eval()
        for param in base_model.parameters():
            param.requires_grad = False
        for param in adapt_model.attention_module.parameters():
            param.requires_grad = False # Attention 모듈은 학습하지 않음
        adapt_model.train()

        train_dataloader = get_pg19_dataloader(tokenizer, batch_size, 'train', max_length)
        eval_dataloader = get_pg19_dataloader(tokenizer, batch_size, 'validation', max_length)

        # 4. 옵티마이저 설정 ✍️
        optimizer = AdamW(adapt_model.parameters(), lr=learning_rate)

        # 5. 증류 학습 트레이너 설정
        trainer = Adapter(
            adapt_model=adapt_model,
            base_model=base_model,
            optimizer=optimizer,
            device=device,
            alpha=alpha,
            temperature=temperature
        )

        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            train_loss = trainer.train_epoch(train_dataloader)
            eval_loss = trainer.eval_epoch(eval_dataloader)
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f}")
        



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
