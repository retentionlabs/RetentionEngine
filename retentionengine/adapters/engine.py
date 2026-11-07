"""Layer trimming for pretrained decoder-only transformers (Llama / Qwen families).

`RetentionEngine` wraps a Hugging Face causal LM, removes a contiguous block of
decoder layers, and keeps the result a standard HF model so it can be saved,
reloaded with `AutoModelForCausalLM.from_pretrained`, and restored with
knowledge distillation (see `retentionengine.utils.DistillationTrainer`).

The Titans memory-module integration (thelethe) is planned for a follow-up PR;
`RetentionEngine.attach_memory_module` raises `NotImplementedError` until then.
"""

import copy
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOTAL_PARAMS = 8_000_000_000
ENGINE_CONFIG_NAME = "retention_engine_config.json"
SIZE_REPORT_NAME = "size_report.json"
# Per-layer config lists that must stay aligned with `num_hidden_layers`.
_PER_LAYER_CONFIG_ATTRS = ("layer_types", "mlp_layer_types")


@dataclass
class RetentionEngineConfig:
    """Trimming request plus a record of what was applied.

    Request fields:
        layers_to_remove: Number of decoder layers to remove explicitly (0 = none).
        removal_start: Index of the first layer of the removed contiguous block.
            ``None`` removes the deepest block that still keeps the final layer,
            i.e. layers ``[L-1-n, L-1)`` for a model with ``L`` layers.
        max_total_params: Parameter budget used when ``auto_resize=True``.
        module_params: Parameters reserved for a future memory module; counted
            against ``max_total_params`` by ``auto_resize``.
        min_layers: Minimum number of layers ``auto_resize`` keeps (>= 1).
        base_model_name: Name/path of the base model (filled in if ``None``).

    Record fields (filled in by `RetentionEngine`, do not set by hand):
        original_num_layers, base_model_params, removed_layer_indices.
    """

    layers_to_remove: int = 0
    removal_start: Optional[int] = None
    max_total_params: int = DEFAULT_MAX_TOTAL_PARAMS
    module_params: int = 0
    min_layers: int = 4
    base_model_name: Optional[str] = None
    original_num_layers: Optional[int] = None
    base_model_params: Optional[int] = None
    removed_layer_indices: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.min_layers < 1:
            raise ValueError(
                f"min_layers must be >= 1 (a model needs a decoder layer), got {self.min_layers}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json_file(self, json_file_path: str) -> None:
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json_file(cls, json_file_path: str) -> "RetentionEngineConfig":
        with open(json_file_path, encoding="utf-8") as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            logger.warning("Ignoring unknown RetentionEngineConfig keys: %s", unknown)
        return cls(**{k: v for k, v in data.items() if k in known})


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the decoder layer list of a Llama/Qwen-style causal LM (``model.model.layers``)."""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise ValueError(
            f"Unsupported model structure for {type(model).__name__}: expected decoder "
            "layers at `model.model.layers` (Llama/Qwen-style causal LM)."
        )
    return layers


def layer_block_indices(
    num_total: int, num_remove: int, start: Optional[int] = None
) -> List[int]:
    """Indices of a contiguous block of ``num_remove`` layers.

    With ``start=None`` the block ends just before the final layer, which is the
    least destructive position for depth pruning (deep layers are the most
    redundant, while the first and last layers matter most).
    """
    if num_remove < 0:
        raise ValueError(f"num_remove must be >= 0, got {num_remove}")
    if num_remove == 0:
        return []
    if num_remove >= num_total:
        raise ValueError(
            f"Cannot remove {num_remove} of {num_total} layers; at least one must remain."
        )
    if start is None:
        start = num_total - 1 - num_remove
    if start < 0 or start + num_remove > num_total:
        raise ValueError(
            f"Removal block [{start}, {start + num_remove}) is out of range for {num_total} layers."
        )
    return list(range(start, start + num_remove))


def remove_layers(model: PreTrainedModel, layer_indices: List[int]) -> PreTrainedModel:
    """Remove decoder layers in place and keep the model internally consistent.

    Updates ``config.num_hidden_layers``, trims per-layer config lists
    (``layer_types``/``mlp_layer_types``), and renumbers every ``layer_idx``
    attribute (e.g. ``self_attn.layer_idx``) so KV-cache slots stay contiguous.
    """
    layers = get_decoder_layers(model)
    num_total = len(layers)
    to_remove = sorted(set(layer_indices))
    if not to_remove:
        return model
    if to_remove[0] < 0 or to_remove[-1] >= num_total:
        raise ValueError(
            f"Layer indices {to_remove} out of range for {num_total} layers."
        )
    if len(to_remove) >= num_total:
        raise ValueError("Cannot remove every decoder layer.")

    config = model.config
    if getattr(config, "is_heterogeneous", False):
        raise NotImplementedError(
            "Layer trimming does not support heterogeneous per-layer configs."
        )

    removed = set(to_remove)
    kept = [i for i in range(num_total) if i not in removed]
    new_layers = nn.ModuleList([layers[i] for i in kept])
    for new_idx, layer in enumerate(new_layers):
        for module in layer.modules():
            if isinstance(getattr(module, "layer_idx", None), int):
                module.layer_idx = new_idx
    model.model.layers = new_layers

    # Keep per-layer config lists aligned with the kept layers.
    for attr in _PER_LAYER_CONFIG_ATTRS:
        values = getattr(config, attr, None)
        if isinstance(values, (list, tuple)) and len(values) == num_total:
            setattr(config, attr, [values[i] for i in kept])
    # Qwen2/Qwen3: layers with index >= max_window_layers use sliding window
    # (older transformers derive this from the index instead of `layer_types`).
    max_window_layers = getattr(config, "max_window_layers", None)
    if isinstance(max_window_layers, int):
        config.max_window_layers = sum(1 for i in kept if i < max_window_layers)
    config.num_hidden_layers = len(new_layers)

    logger.info(
        "Removed layers %s: %d -> %d layers", to_remove, num_total, len(new_layers)
    )
    return model


class ModelSizeManager:
    """Utility class for model size calculation and management"""

    @staticmethod
    def count_parameters(model: nn.Module, exclude_embeddings: bool = False) -> int:
        """Count unique parameters (tied weights are counted once)."""
        total = 0
        for name, param in model.named_parameters():
            if exclude_embeddings and (
                "embed" in name.lower() or "lm_head" in name.lower()
            ):
                continue
            total += param.numel()
        return total

    @staticmethod
    def params_per_layer(model: PreTrainedModel) -> int:
        """Actual parameter count of one decoder layer."""
        layers = get_decoder_layers(model)
        return sum(p.numel() for p in layers[0].parameters())

    @staticmethod
    def calculate_layers_to_remove(
        base_model: PreTrainedModel,
        module_params: int,
        max_total_params: float = DEFAULT_MAX_TOTAL_PARAMS,
        min_layers: int = 4,
    ) -> int:
        """Number of layers to remove so that base + module params fit the budget."""
        if min_layers < 1:
            raise ValueError(f"min_layers must be >= 1, got {min_layers}")
        total = ModelSizeManager.count_parameters(base_model) + module_params
        if total <= max_total_params:
            return 0
        excess = total - max_total_params
        needed = math.ceil(excess / ModelSizeManager.params_per_layer(base_model))
        max_removable = max(0, len(get_decoder_layers(base_model)) - min_layers)
        if needed > max_removable:
            logger.warning(
                "Need %d layers removed to fit %.2fB params but only %d can go (min_layers=%d).",
                needed,
                max_total_params / 1e9,
                max_removable,
                min_layers,
            )
        return min(needed, max_removable)


class RetentionEngine(nn.Module):
    """A trimmed Hugging Face causal LM.

    Args:
        basemodel: Llama/Qwen-style ``*ForCausalLM`` model. It is trimmed in place.
        config: Trimming request; defaults to no trimming.
        auto_resize: Derive the number of removed layers from
            ``max_total_params``/``module_params`` instead of ``layers_to_remove``.
            Off by default so layers are never pruned silently; trimming must be
            requested explicitly.
    """

    def __init__(
        self,
        basemodel: PreTrainedModel,
        config: Optional[RetentionEngineConfig] = None,
        auto_resize: bool = False,
    ):
        super().__init__()
        config = (
            copy.deepcopy(config) if config is not None else RetentionEngineConfig()
        )
        if config.base_model_name is None:
            config.base_model_name = (
                getattr(basemodel.config, "_name_or_path", None) or None
            )

        if config.removed_layer_indices:
            self._check_already_trimmed(basemodel, config)
        else:
            self._trim(basemodel, config, auto_resize)

        self.model = basemodel
        self.config = config
        self.auto_resize = auto_resize

    @staticmethod
    def _check_already_trimmed(
        basemodel: PreTrainedModel, config: RetentionEngineConfig
    ) -> None:
        """Validate a model that was trimmed before (e.g. loaded via `from_pretrained`)."""
        current = getattr(basemodel.config, "num_hidden_layers", None)
        if config.original_num_layers is None:
            raise ValueError(
                "`removed_layer_indices` is a record filled in by RetentionEngine. To trim a "
                "model, set `layers_to_remove`/`removal_start` or call `remove_layers()`."
            )
        expected = config.original_num_layers - len(config.removed_layer_indices)
        if current != expected:
            raise ValueError(
                f"Config records a trimmed model with {expected} layers, but the given model "
                f"has {current}. Pass the trimmed model or a fresh RetentionEngineConfig."
            )

    def _trim(
        self,
        basemodel: PreTrainedModel,
        config: RetentionEngineConfig,
        auto_resize: bool,
    ) -> None:
        config.original_num_layers = getattr(
            basemodel.config, "num_hidden_layers", None
        )
        config.base_model_params = ModelSizeManager.count_parameters(basemodel)

        num_remove = config.layers_to_remove
        if auto_resize:
            if num_remove:
                raise ValueError(
                    "Set either `layers_to_remove` or `auto_resize=True`, not both."
                )
            num_remove = self._layers_for_budget(basemodel, config)
        if num_remove == 0:
            return

        num_total = len(get_decoder_layers(basemodel))
        indices = layer_block_indices(num_total, num_remove, config.removal_start)
        remove_layers(basemodel, indices)
        config.layers_to_remove = num_remove
        config.removed_layer_indices = indices

    @staticmethod
    def _layers_for_budget(
        basemodel: PreTrainedModel, config: RetentionEngineConfig
    ) -> int:
        max_total = getattr(config, "max_total_params", DEFAULT_MAX_TOTAL_PARAMS)
        module_params = getattr(config, "module_params", 0) or 0
        total = config.base_model_params + module_params
        logger.info(
            "Base %.2fB + module %.2fB = %.2fB params (budget %.2fB)",
            config.base_model_params / 1e9,
            module_params / 1e9,
            total / 1e9,
            max_total / 1e9,
        )
        return ModelSizeManager.calculate_layers_to_remove(
            basemodel, module_params, max_total, getattr(config, "min_layers", 4)
        )

    def attach_memory_module(self, *args, **kwargs):
        """Attach a Titans memory module (thelethe). Planned for a follow-up PR."""
        raise NotImplementedError(
            "Titans memory-module integration is not available yet: thelethe's "
            "PreTrainedTitansModel/CronosForCausalLM are still empty upstream. This release "
            "covers layer trimming + restoration distillation only."
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def get_model_size_info(self) -> Dict[str, Any]:
        """Parameter counts of the wrapped model (each parameter counted once)."""
        base_params = ModelSizeManager.count_parameters(self.model)
        total_params = ModelSizeManager.count_parameters(self)
        module_params = total_params - base_params
        return {
            "base_model_params": base_params,
            "module_params": module_params,
            "reserved_module_params": self.config.module_params,
            "total_params": total_params,
            "total_params_billion": total_params / 1e9,
            "original_base_model_params": self.config.base_model_params,
            "layers_removed": len(self.config.removed_layer_indices),
            "removed_layer_indices": list(self.config.removed_layer_indices),
            "original_layers": self.config.original_num_layers,
            "current_layers": getattr(self.model.config, "num_hidden_layers", None),
        }

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        """Save the trimmed model in standard HF format plus the engine config.

        ``save_directory`` can be loaded with either ``RetentionEngine.from_pretrained``
        or plain ``AutoModelForCausalLM.from_pretrained``.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory, **kwargs)
        self.config.to_json_file(os.path.join(save_directory, ENGINE_CONFIG_NAME))
        with open(
            os.path.join(save_directory, SIZE_REPORT_NAME), "w", encoding="utf-8"
        ) as f:
            json.dump(self.get_model_size_info(), f, indent=2)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **model_kwargs
    ) -> "RetentionEngine":
        """Load a model saved by `save_pretrained` (layers are not trimmed again).

        ``model_kwargs`` go to ``AutoModelForCausalLM.from_pretrained`` (e.g. ``torch_dtype``).
        The engine config is read from a local directory; without it the model is
        wrapped as-is.
        """
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **model_kwargs
        )
        config_path = os.path.join(pretrained_model_name_or_path, ENGINE_CONFIG_NAME)
        config = (
            RetentionEngineConfig.from_json_file(config_path)
            if os.path.isfile(config_path)
            else None
        )
        return cls(model, config)
