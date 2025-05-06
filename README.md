# RetentionEngine
A simple adapter implementation to transform pretrained Transformer-family models into the Titans architecture.

## Experiments Setup
### Exp 1: Comparison with Base Models [3B]
#### Baseline
- Titans 4B (Gemma3 4B IT)
- Llama3.2 3B Instruct
- Qwen3 4B
- Gemma3 4B IT
- RAG + Llama3.2 3B Instruct
- RAG + Qwen3 4B
- RAG + Gemma3 4B IT

#### Suggested
- Atlas-L 3B (Llama3.2 3B Instruct)
- Atlas-Q 4B (Qwen3 4B)
- Atlas-G 4B (Gemma3 4B IT)

### Exp 2: Comparison with Base Models [8B]
#### Baseline
- Titans 9B (Gemma2 9B IT)
- Llama3.1 8B Instruct
- Qwen3 8B
- Gemma3 9B IT
- RAG + Llama3.1 8B Instruct
- RAG + Qwen3 8B
- RAG + Gemma3 9B IT

#### Suggested
- Atlas-L 8B (Llama3.1 8B Instruct)
- Atlas-Q 8B (Qwen3 8B)
- Atlas-G 9B (Gemma3 9B IT)

### Exp 3: Comparison with Recurrent Models [8B]
#### Baseline
- Titans 9B (Gemma2 9B IT)
- Mamba2 8B
- RWKV6 7B

#### Suggested
- Atlas-L 3B (Llama3.2 3B Instruct)
- Atlas-Q 4B (Qwen3 4B)
- Atlas-G 4B (Gemma3 4B IT)
- Atlas-L 8B (Llama3.1 8B Instruct)
- Atlas-Q 8B (Qwen3 8B)
- Atlas-G 9B (Gemma3 9B IT)
