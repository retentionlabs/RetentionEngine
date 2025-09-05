from thelethe.titans import PretrainedTitansConfig, PreTrainedTitansModel
from transformers import PreTrainedModel, AutoTokenizer, AutoConfig
from torch import nn
from torch.optim import AdamW
import torch
from typing import Optional, Dict, Any
import math


class RetentionEngineConfig(PretrainedTitansConfig):
    """Extended configuration with size management"""
    
    def __init__(
        self,
        max_total_params: float = 8e9,  # 8B
        base_model_name: str = None,
        base_model_params: float = None,
        layers_to_remove: int = 0,
        original_num_layers: int = None,
        module_params: float = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_total_params = max_total_params
        self.base_model_name = base_model_name
        self.base_model_params = base_model_params
        self.layers_to_remove = layers_to_remove
        self.original_num_layers = original_num_layers
        self.module_params = module_params


class ModelSizeManager:
    """Utility class for model size calculation and management"""
    
    @staticmethod
    def count_parameters(model: nn.Module, exclude_embeddings: bool = False) -> int:
        """Count total parameters in a model"""
        total = 0
        for name, param in model.named_parameters():
            if exclude_embeddings and ('embed' in name.lower() or 'lm_head' in name.lower()):
                continue
            total += param.numel()
        return total
    
    @staticmethod
    def estimate_layer_params(model: PreTrainedModel) -> int:
        """Estimate parameters per transformer layer"""
        config = model.config
        
        # Get model architecture details
        hidden_size = getattr(config, 'hidden_size', 4096)
        intermediate_size = getattr(config, 'intermediate_size', hidden_size * 4)
        num_attention_heads = getattr(config, 'num_attention_heads', 32)
        num_key_value_heads = getattr(config, 'num_key_value_heads', num_attention_heads)
        
        # Estimate parameters per layer
        # Self-attention: Q, K, V, O projections
        attn_params = (
            hidden_size * hidden_size +  # Q
            hidden_size * (hidden_size * num_key_value_heads // num_attention_heads) * 2 +  # K, V
            hidden_size * hidden_size  # O
        )
        
        # MLP: gate, up, down projections
        mlp_params = (
            hidden_size * intermediate_size * 2 +  # gate, up
            intermediate_size * hidden_size  # down
        )
        
        # Layer norms
        norm_params = hidden_size * 2 * 2  # 2 layer norms, weight + bias
        
        return attn_params + mlp_params + norm_params
    
    @staticmethod
    def calculate_layers_to_remove(
        base_model: PreTrainedModel,
        module_params: int,
        max_total_params: float = 8e9
    ) -> int:
        """Calculate how many layers to remove to stay under size limit"""
        
        total_base_params = ModelSizeManager.count_parameters(base_model)
        
        if total_base_params + module_params <= max_total_params:
            return 0
        
        # Need to remove layers
        excess_params = (total_base_params + module_params) - max_total_params
        params_per_layer = ModelSizeManager.estimate_layer_params(base_model)
        
        layers_to_remove = math.ceil(excess_params / params_per_layer)
        
        # Ensure we don't remove too many layers
        num_layers = getattr(base_model.config, 'num_hidden_layers', 32)
        max_removable = max(0, num_layers - 4)  # Keep at least 4 layers
        
        return min(layers_to_remove, max_removable)


class RetentionEngine(nn.Module):
    def __init__(
        self, 
        basemodel: PreTrainedModel, 
        config: RetentionEngineConfig,
        auto_resize: bool = True
    ):
        super().__init__()
        
        self.original_basemodel = basemodel
        self.auto_resize = auto_resize
        
        self.module = PreTrainedTitansModel(config)
        
        module_params = ModelSizeManager.count_parameters(self.module)
        config.module_params = module_params
        
        if auto_resize:
            basemodel = self._resize_basemodel(basemodel, config, module_params)
        
        self.module.model = basemodel
        self._attach_embeddings()
        
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
        
    def _resize_basemodel(
        self, 
        basemodel: PreTrainedModel, 
        config: RetentionEngineConfig,
        module_params: int
    ) -> PreTrainedModel:
        """Resize base model by removing layers if necessary"""
        
        # Calculate base model size
        base_params = ModelSizeManager.count_parameters(basemodel)
        config.base_model_params = base_params
        config.original_num_layers = getattr(basemodel.config, 'num_hidden_layers', None)
        
        # Check if resizing is needed
        total_params = base_params + module_params
        
        print(f"Base model params: {base_params/1e9:.2f}B")
        print(f"Module params: {module_params/1e9:.2f}B")
        print(f"Total params: {total_params/1e9:.2f}B")
        
        if total_params <= config.max_total_params:
            print(f"Model size is within limit ({config.max_total_params/1e9:.1f}B). No resizing needed.")
            config.layers_to_remove = 0
            return basemodel
        
        # Calculate layers to remove
        layers_to_remove = ModelSizeManager.calculate_layers_to_remove(
            basemodel, 
            module_params,
            config.max_total_params
        )
        
        if layers_to_remove == 0:
            print("Cannot reduce model size further while maintaining minimum layers.")
            return basemodel
        
        print(f"Removing {layers_to_remove} layers to meet size constraint...")
        config.layers_to_remove = layers_to_remove
        
        # Remove layers from model
        resized_model = self._remove_layers(basemodel, layers_to_remove)
        
        # Recalculate size
        new_base_params = ModelSizeManager.count_parameters(resized_model)
        new_total = new_base_params + module_params
        print(f"Resized base model: {new_base_params/1e9:.2f}B")
        print(f"New total size: {new_total/1e9:.2f}B")
        
        return resized_model
    
    def _remove_layers(self, model: PreTrainedModel, num_layers: int) -> PreTrainedModel:
        """Remove specified number of layers from the front of the model"""
        
        # Access the transformer layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            print("Warning: Could not identify layer structure. Returning original model.")
            return model
        
        total_layers = len(layers)
        if num_layers >= total_layers - 4:
            print(f"Warning: Cannot remove {num_layers} layers. Keeping minimum 4 layers.")
            num_layers = max(0, total_layers - 4)
            if num_layers == 0:
                return model
        
        new_layers = nn.ModuleList(layers[num_layers:])
        
        # Replace layers in model
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            model.model.layers = new_layers
        
        # Update config
        if hasattr(model.config, 'num_hidden_layers'):
            model.config.num_hidden_layers = len(new_layers)
        elif hasattr(model.config, 'n_layer'):
            model.config.n_layer = len(new_layers)
        
        print(f"Layers reduced from {total_layers} to {len(new_layers)}")
        
        return model
    
    def _merge_configs(self, base_config):
        """Merge base model configuration with module configuration"""
        if hasattr(base_config, 'to_dict'):
            config_dict = base_config.to_dict()
            for key, value in config_dict.items():
                if key not in ['num_hidden_layers', 'n_layer']:  # Don't override layer count
                    self.config.__dict__[key] = value
        elif hasattr(base_config, '__dict__'):
            for key, value in base_config.__dict__.items():
                if not key.startswith('_') and key not in ['num_hidden_layers', 'n_layer']:
                    self.config.__dict__[key] = value
    
    def _attach_embeddings(self):
        """Attach the input embeddings from the base model to the adapter model"""
        if self.module.model is not None and hasattr(self.module.model, 'get_input_embeddings'):
            input_embeddings = self.module.model.get_input_embeddings()
            self.module.set_input_embeddings(input_embeddings)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def get_model_size_info(self) -> Dict[str, Any]:
        """Get detailed size information about the model"""
        base_params = ModelSizeManager.count_parameters(self.module.model) if self.module.model else 0
        module_params = ModelSizeManager.count_parameters(self.module, exclude_embeddings=True)
        total_params = base_params + module_params
        
        return {
            'base_model_params': base_params,
            'base_model_size_gb': base_params / 1e9,
            'module_params': module_params,
            'module_size_gb': module_params / 1e9,
            'total_params': total_params,
            'total_size_gb': total_params / 1e9,
            'layers_removed': self.config.layers_to_remove,
            'original_layers': self.config.original_num_layers,
            'current_layers': getattr(self.module.model.config, 'num_hidden_layers', None) if self.module.model else None
        }
    
    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        basemodel: PreTrainedModel,
        auto_resize: bool = True,
        max_total_params: float = 8e9
    ) -> "RetentionEngine":
        """Load a RetentionEngine from a pretrained model and configuration"""
        
        # Load config with size constraints
        config = RetentionEngineConfig.from_pretrained(model_id)
        config.max_total_params = max_total_params
        config.base_model_name = getattr(basemodel.config, '_name_or_path', 'unknown')
        
        # Create engine with auto-resizing
        engine = cls(basemodel, config, auto_resize=auto_resize)
        
        # Load pretrained module weights
        module = PreTrainedTitansModel.from_pretrained(model_id)
        engine.module.model = engine.module.model  # Use the resized model
        engine.module = module
                
        return engine
    
    def save_pretrained(self, save_directory: str):
        """Save the RetentionEngine to a directory"""
        # Save size information in config
        size_info = self.get_model_size_info()
        for key, value in size_info.items():
            setattr(self.config, f'size_{key}', value)
        
        self.module.save_pretrained(save_directory)
        self.config.save_pretrained(save_directory)
        
        # Save size report
        import json
        with open(f"{save_directory}/size_report.json", 'w') as f:
            json.dump(size_info, f, indent=2)


# Example usage
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    
    # Load a base model (e.g., 13B model)
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-13b-hf",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Create config with 8B size limit
    config = RetentionEngineConfig(
        max_total_params=8e9,  # 8B max
        base_model_name="meta-llama/Llama-2-13b-hf",
        # Add other Titans config parameters here
    )
    
    # Create engine with automatic resizing
    engine = RetentionEngine(
        basemodel=base_model,
        config=config,
        auto_resize=True  # Automatically remove layers if needed
    )
    
    # Check final size
    size_info = engine.get_model_size_info()
    print("\nFinal Model Size Information:")
    for key, value in size_info.items():
        print(f"  {key}: {value}")
    
    # Save the resized model
    engine.save_pretrained("./retention_engine_8b")