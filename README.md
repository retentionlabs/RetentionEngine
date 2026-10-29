# RetentionEngine
Trim layers of pretrained Transformer models (Llama / Qwen families) and restore their quality with knowledge distillation.

## Scope
- **Layer trimming** (`RetentionEngine`): removes a contiguous block of decoder layers — by default the deep block just before the final layer (`removal_start` picks another position). The trimmed model stays a standard Hugging Face model (`save_pretrained` / `AutoModelForCausalLM.from_pretrained`). Pruning by parameter budget is opt-in (`auto_resize=True`).
- **Restoration KD** (`retentionengine.utils.DistillationTrainer`): `loss = (1 - α)·LM + α·T²·KL(teacher ‖ student)` over non-padding tokens. The teacher runs in bf16 by default; FP8 (Transformer Engine) is opt-in with `use_fp8_teacher=True`.
- **Titans memory module** (thelethe): planned for a follow-up PR. `RetentionEngine.attach_memory_module()` raises `NotImplementedError`; nothing in the package imports thelethe yet. The experiment plan below depends on it.

## Install
```bash
git clone https://github.com/retentionlabs/RetentionEngine.git
cd RetentionEngine
uv venv && uv pip install -e ".[train]"
```

| Extra | Installs | Needed for |
|---|---|---|
| `train` | datasets, wandb, accelerate | restoration training / notebooks |
| `fp8` | transformer-engine[pytorch] | FP8 teacher (`use_fp8_teacher=True`, CUDA GPU) |
| `datagen` | json-repair, accelerate | `scripts/dataset_generator.py` |
| `titans` | thelethe (pinned git commit) | future Titans integration (not used yet) |

## Usage
```python
import torch
from transformers import AutoModelForCausalLM
from retentionengine import RetentionEngine, RetentionEngineConfig

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16)
engine = RetentionEngine(base, RetentionEngineConfig(layers_to_remove=4))  # drops layers 31-34 of 36
engine.save_pretrained("pretrained/Qwen3-8B-trimmed")
```
1. `usage.ipynb` — trim and save the student.
2. `python scripts/dataset_generator.py --model-name Qwen/Qwen3-8B --output data/qwen3_memory_aligned_dataset.jsonl` — generate restoration data.
3. `restoration_training.ipynb` — distill the full model into the trimmed one and compare their generations.

## Tests
CPU-only smoke tests with tiny random Llama/Qwen3 models:
```bash
uv run pytest
```

## Experiments Setup (planned, needs the Titans integration)
### Exp 1: Comparison with Base Models [3B]
#### Baseline
- Titans 4B (Gemma3 4B IT)
- Atlas 4B (Gemma3 4B IT)
- Llama3.2 3B Instruct
- Qwen3 4B
- Gemma3 4B IT
- RAG + Llama3.2 3B Instruct
- RAG + Qwen3 4B
- RAG + Gemma3 4B IT

#### Suggested
- Cronos-L 3B (Llama3.2 3B Instruct)
- Cronos-Q 4B (Qwen3 4B)
- Cronos-G 4B (Gemma3 4B IT)

### Exp 2: Comparison with Base Models [8B]
#### Baseline
- Titans 9B (Gemma2 9B IT)
- Atlas 9B (Gemma2 9B IT)
- Llama3.1 8B Instruct
- Qwen3 8B
- Gemma3 9B IT
- RAG + Llama3.1 8B Instruct
- RAG + Qwen3 8B
- RAG + Gemma3 9B IT

#### Suggested
- Cronos-L 8B (Llama3.1 8B Instruct)
- Cronos-Q 8B (Qwen3 8B)
- Cronos-G 9B (Gemma3 9B IT)

### Exp 3: Comparison with Recurrent Models [8B]
#### Baseline
- Titans 9B (Gemma2 9B IT)
- Atlas 9B (Gemma2 9B IT)
- Mamba2 8B
- RWKV6 7B

#### Suggested
- Cronos-L 3B (Llama3.2 3B Instruct)
- Cronos-Q 4B (Qwen3 4B)
- Cronos-G 4B (Gemma3 4B IT)
- Cronos-L 8B (Llama3.1 8B Instruct)
- Cronos-Q 8B (Qwen3 8B)
- Cronos-G 9B (Gemma3 9B IT)


## Deploy
PyPI rejects direct-URL dependencies, so drop or replace the `titans` extra before publishing.
```bash
rm -rf dist/
rm -rf build/
rm -rf *.egg-info

python -m pip install --upgrade pip
pip install flit
python -m flit build

echo "Uploading..."
python -m flit publish

echo "Deployment complete!"
```
