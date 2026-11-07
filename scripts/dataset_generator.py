# -*- coding: utf-8 -*-
"""
Memory-aligned restoration dataset generator.
----------------------------------------------------------------------
Generates synthetic samples with an instruction-tuned causal LM and writes them
as JSONL (one sample per line); about 2/3 Korean and 1/3 English samples.

Usage:
    pip install 'retentionengine[datagen]'
    python scripts/dataset_generator.py \
        --model-name Qwen/Qwen3-8B \
        --output data/qwen3_memory_aligned_dataset.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
import time
import uuid
from collections import Counter
from typing import Dict, List, Optional

import json_repair  # Repairs malformed JSON; install via: pip install json-repair
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_OUTPUT = "data/qwen3_memory_aligned_dataset.jsonl"
SYSTEM_PROMPT = (
    "You are a helpful assistant that generates high-quality synthetic data."
)
REQUIRED_KEYS = ("input", "target_output")


def make_seed_prompt(batch_size: int = 25) -> str:  # Reduced default for faster testing
    """
    Return a Korean prompt with 2/3 probability, and an English prompt with 1/3 probability.
    """
    if random.random() < 0.33:  # Approx. 1/3 probability for English
        return make_seed_prompt_english(batch_size)
    else:
        return make_seed_prompt_korean(batch_size)


def make_seed_prompt_korean(batch_size: int = 25) -> str:
    """
    Korean prompt
    """
    domains = [
        ("math", 20),
        ("cs", 20),
        ("physics", 15),
        ("literature", 15),
        ("philosophy", 10),
        ("history", 10),
        ("biology", 5),
        ("economics", 5),
    ]
    domain_str = ", ".join([f"{d}({p}%)" for d, p in domains])

    prompt = f"""
다음 JSON 스키마를 따르는 데이터를 정확히 {batch_size}개 포함하는 단일 JSON 배열(array)을 생성해줘.

{{
  "id": "sample_001",
  "domain": "예: math, physics, literature, cs, philosophy 등",
  "input": "장문 또는 단문 입력. 다양한 도메인 혼합 가능",
  "target_output": "모델이 생성할 것으로 예상되는 출력",
  "reasoning_trace": [
    {{
      "step": 1,
      "thought": "내부 사고 과정",
      "memory_read": ["참조한 메모리 블록 이름"],
      "inference": "추론 내용"
    }}
  ],
  "memory_access_pattern": {{
    "reads": ["메모리_블록_A", "메모리_블록_B"],
    "writes": ["결과_요약_01"]
  }},
  "metadata": {{
    "context_length": 입력 토큰 수,
    "reasoning_depth": "trivial / intermediate / advanced"
  }}
}}

### 생성 규칙:
- 30%: 짧은 질문 (100~500 토큰), 출력 분포 정렬용
- 50%: 중간 길이 CoT (1K~5K 토큰), 메모리 트레이스 포함
- 20%: 장문 크로스 도메인 (8K~30K 토큰), multi-hop 추론
- 도메인 분포: {domain_str}
- reasoning_trace는 2~5단계 포함
- memory_read/write는 실제 메모리 블록 이름 사용 (예: physics_laws_v1, literature_themes_02)
- 문자열 내부에 따옴표(\")가 있다면 반드시 백슬래시(\\)로 이스케이프 처리해줘 (예: `\"`).
- 오직 하나의 완성된 JSON 배열만 출력해줘. 설명, 마크다운, ```json``` 태그는 절대 포함하지 마.

### 완벽한 출력 예시:
[
  {{
    "id": "temp_001",
    "domain": "literature",
    "input": "소설 '데미안'의 주제는 무엇인가?",
    "target_output": "헤르만 헤세의 소설 '데미안'의 핵심 주제는 개인의 자아 발견과 성장 과정입니다. 주인공 싱클레어가 알을 깨고 나오듯, 기존의 세계를 벗어나 자기 자신만의 길을 찾아가는 과정을 그립니다.",
    "reasoning_trace": [],
    "memory_access_pattern": {{"reads": [], "writes": []}},
    "metadata": {{"context_length": 50, "reasoning_depth": "trivial"}}
  }}
]
"""
    return prompt.strip()


def make_seed_prompt_english(batch_size: int = 25) -> str:
    """
    English prompt (called with 1/3 probability)
    """
    domains = [
        ("math", 20),
        ("cs", 20),
        ("physics", 15),
        ("literature", 15),
        ("philosophy", 10),
        ("history", 10),
        ("biology", 5),
        ("economics", 5),
    ]
    domain_str = ", ".join([f"{d}({p}%)" for d, p in domains])

    prompt = f"""
Generate a single JSON array containing exactly {batch_size} samples that follow the JSON schema below.

{{
  "id": "sample_001",
  "domain": "e.g., math, physics, literature, cs, philosophy",
  "input": "Long or short input. Mix of domains allowed",
  "target_output": "Output expected from the model",
  "reasoning_trace": [
    {{
      "step": 1,
      "thought": "Internal reasoning process",
      "memory_read": ["memory_block_name"],
      "inference": "Inference content"
    }}
  ],
  "memory_access_pattern": {{
    "reads": ["memory_block_A", "memory_block_B"],
    "writes": ["summary_01"]
  }},
  "metadata": {{
    "context_length": "number of input tokens",
    "reasoning_depth": "trivial / intermediate / advanced"
  }}
}}

### Rules:
- 30%: Short questions (100-500 tokens), for output distribution alignment
- 50%: Medium-length CoT (1K-5K tokens), with memory trace
- 20%: Long cross-domain (8K-30K tokens), multi-hop reasoning
- Domain distribution: {domain_str}
- reasoning_trace must have 2-5 steps
- Use realistic memory block names (e.g., physics_laws_v1, literature_themes_02)
- Output only a single, complete JSON array. Do not include any explanation, markdown, or ```json``` tags.
- Example: `[ {{"id": "...", ...}} ]`
"""
    return prompt.strip()


def is_valid_sample(item) -> bool:
    """A sample needs a non-empty string for every key in ``REQUIRED_KEYS``."""
    return isinstance(item, dict) and all(
        isinstance(item.get(key), str) and item[key].strip() for key in REQUIRED_KEYS
    )


def json_structure_closed(text: str) -> bool:
    """True if every ``[``/``{`` opened outside a string is closed and no string is left open.

    Output cut off mid-array fails this; json_repair would otherwise "complete" it
    into plausible-looking but truncated samples.
    """
    depth = 0
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not in_string


def parse_json_array(text: str, stats: Optional[Counter] = None) -> List[Dict]:
    """
    Extract the valid samples (see ``is_valid_sample``) of a JSON array in the generated text.

    json_repair is only tried when strict parsing fails and the array is structurally
    closed (``json_structure_closed``); truncated output is rejected, not repaired.
    ``stats`` (optional) counts ``repaired`` samples (kept from a repaired array),
    ``rejected`` samples (failed validation) and ``rejected_batches``.
    """
    stats = Counter() if stats is None else stats
    # Find the first '[' and the last ']' to capture the array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        print("⚠️ WARNING: Could not find a JSON array in the response.")
        stats["rejected_batches"] += 1
        return []

    json_array_str = match.group(0)
    repaired = False
    try:
        data = json.loads(json_array_str)
    except json.JSONDecodeError:
        if not json_structure_closed(json_array_str):
            print("⚠️ WARNING: JSON array is not closed (truncated?); not repairing.")
            stats["rejected_batches"] += 1
            return []
        print("⚠️ WARNING: JSON decoding failed. Attempting to repair with json_repair.")
        try:
            data = json_repair.loads(json_array_str)
        except Exception as e:
            print(f"🆘 CRITICAL: Repair with json_repair also failed: {e}")
            stats["rejected_batches"] += 1
            return []
        repaired = True

    if not isinstance(data, list):
        print("⚠️ WARNING: Parsed data is not a list.")
        stats["rejected_batches"] += 1
        return []

    valid_items = [item for item in data if is_valid_sample(item)]
    stats["rejected"] += len(data) - len(valid_items)
    if repaired:
        stats["repaired"] += len(valid_items)
    return valid_items


def build_prompt_inputs(tokenizer, prompt: str):
    """Tokenize a single-turn chat with the tokenizer's own chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    # `enable_thinking=False` turns off Qwen3's <think> block; other templates ignore it.
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )


def generate_dataset(
    model_name: str = DEFAULT_MODEL_NAME,
    output_file: str = DEFAULT_OUTPUT,
    target_count: int = 10000,
    batch_size: int = 25,
    max_new_tokens: int = 16000,
    max_retries: int = 10,
):
    """
    Generate a dataset using the specified model and save it to a JSONL file.

    Stops early if ``max_retries`` consecutive batches yield no valid samples.
    A generation that stops at ``max_new_tokens`` without EOS is truncated and is
    retried rather than repaired. Raises ``RuntimeError`` if the model or
    tokenizer cannot be loaded.
    """
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"🚀 Loading model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )
        print(f"✅ Model loaded successfully. Output file: {output_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name!r}: {e}") from e

    context_length = getattr(model.config, "max_position_embeddings", 32768)
    generated_samples = []
    consecutive_failures = 0
    stats = Counter()

    with open(output_file, "w", encoding="utf-8") as f_out:
        while len(generated_samples) < target_count:
            if consecutive_failures >= max_retries:
                print(f"🆘 Giving up after {max_retries} consecutive failed batches.")
                break

            remaining = target_count - len(generated_samples)
            current_batch_size = min(batch_size, remaining)
            print(
                f"🔄 Generating: {len(generated_samples)} / {target_count} | Current batch size: {current_batch_size}"
            )

            prompt = make_seed_prompt(current_batch_size)
            inputs = build_prompt_inputs(tokenizer, prompt).to(model.device)

            # Respect the model's context window
            input_len = inputs["input_ids"].shape[1]
            max_new = min(max_new_tokens, context_length - input_len - 10)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        temperature=0.5,
                        do_sample=True,
                        top_p=0.9,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                # Decode only the newly generated tokens
                generated_tokens = outputs[0][input_len:]
                # No final EOS: generation stopped at max_new_tokens and the JSON is
                # truncated. Retry instead of letting json_repair complete it.
                if (
                    generated_tokens.numel() == 0
                    or generated_tokens[-1].item() != tokenizer.eos_token_id
                ):
                    stats["truncated_batches"] += 1
                    consecutive_failures += 1
                    print(
                        f"⚠️ Output hit max_new_tokens={max_new} without EOS (truncated); discarded ({consecutive_failures}/{max_retries}). Retrying after a delay."
                    )
                    time.sleep(5)
                    continue

                content = tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                ).strip()

                new_samples = parse_json_array(content, stats)
                print(f"✅ Extracted {len(new_samples)} valid samples from this batch.")

                if not new_samples:
                    consecutive_failures += 1
                    print(
                        f"⚠️ Failed to generate valid samples in this batch ({consecutive_failures}/{max_retries}). Retrying after a delay."
                    )
                    time.sleep(5)
                    continue

                consecutive_failures = 0
                for sample in new_samples:
                    if len(generated_samples) >= target_count:
                        break
                    sample["id"] = str(uuid.uuid4())  # Assign a unique ID
                    f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    generated_samples.append(sample)

                f_out.flush()  # Write data to disk periodically
                print(f"📊 Total generated: {len(generated_samples)} / {target_count}")

                time.sleep(2)  # Small delay between batches
            except Exception as e:
                consecutive_failures += 1
                print(
                    f"🆘 ERROR during generation ({consecutive_failures}/{max_retries}): {e}. Retrying after delay."
                )
                time.sleep(5)
                continue

    print("\n🎉 Generation finished.")
    print(f"📁 Data saved to: {output_file}")
    print(f"📊 Total samples generated: {len(generated_samples)}")
    print(
        f"🧾 Samples kept from json_repair'd arrays: {stats['repaired']} | "
        f"samples rejected by validation: {stats['rejected']} | "
        f"truncated generations: {stats['truncated_batches']} | "
        f"other rejected batches: {stats['rejected_batches']}"
    )
    return generated_samples


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Generate a memory-aligned restoration dataset (JSONL)."
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="HF model used to generate samples.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument(
        "--target-count", type=int, default=5000, help="Number of samples to generate."
    )
    parser.add_argument(
        "--batch-size", type=int, default=25, help="Samples requested per prompt."
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=16000, help="Generation cap per prompt."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Stop after this many consecutive failed batches.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for prompt language selection.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns 1 (failure) when no sample was written."""
    args = parse_args(argv)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    samples = generate_dataset(
        model_name=args.model_name,
        output_file=args.output,
        target_count=args.target_count,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
    )
    if not samples:
        print(f"🆘 ERROR: No samples were written to {args.output}.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
