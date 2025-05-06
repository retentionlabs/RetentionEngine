from thelethe.titans import PretrainedTitansConfig, PreTrainedTitansModel
from transformers import PreTrainedModel
from torch import nn


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

    def adapt(self, *args, **kwargs):
        pass

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
