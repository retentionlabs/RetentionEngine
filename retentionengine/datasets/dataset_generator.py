# -*- coding: utf-8 -*-
"""
Qwen2-based Memory Module Training Dataset Generator (Stabilized Version)
----------------------------------------------------------------------
Generated data: RETENTIONENGINE/retentionengine/datasets/qwen2_memory_aligned_dataset.jsonl
→ Includes 2/3 Korean and 1/3 English samples.
"""

import json
import re
import time
import random
import uuid
from typing import List, Dict
import os  # Added for directory creation
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json_repair  # Library for repairing malformed JSON; install via: pip install json-repair

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
        ("math", 20), ("cs", 20), ("physics", 15), ("literature", 15),
        ("philosophy", 10), ("history", 10), ("biology", 5), ("economics", 5)
    ]
    domain_str = ", ".join([f"{d}({p}%)" for d, p in domains])

    prompt = f"""
다음 JSON 스키마를 따르는 데이터를 정확히 {batch_size}개 포함하는 단일 JSON 배열(array)을 생성해줘.

{{
  "id": "sample_001",
  "domain": "예: math, physics, literature, cs, philosophy 등",
  "input": "장문 또는 단문 입력. 다양한 도메인 혼합 가능",
  "target_output": "Qwen2이 생성할 것으로 예상되는 출력",  # Updated to Qwen2
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
        ("math", 20), ("cs", 20), ("physics", 15), ("literature", 15),
        ("philosophy", 10), ("history", 10), ("biology", 5), ("economics", 5)
    ]
    domain_str = ", ".join([f"{d}({p}%)" for d, p in domains])

    prompt = f"""
Generate a single JSON array containing exactly {batch_size} samples that follow the JSON schema below.

{{
  "id": "sample_001",
  "domain": "e.g., math, physics, literature, cs, philosophy",
  "input": "Long or short input. Mix of domains allowed",
  "target_output": "Output expected from Qwen2",  # Updated to Qwen2
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

def parse_json_array(text: str) -> List[Dict]:
    """
    Extract a valid JSON array from the generated text.
    """
    # Find the first '[' and the last ']' to capture the array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        print("⚠️ WARNING: Could not find a JSON array in the response.")
        return []

    json_array_str = match.group(0)
    try:
        data = json.loads(json_array_str)
    except json.JSONDecodeError:
        # If standard JSON decoding fails, attempt to repair with json_repair
        print("⚠️ WARNING: JSON decoding failed. Attempting to repair with json_repair.")
        try:
            data = json_repair.loads(json_array_str)
        except Exception as e:
            print(f"🆘 CRITICAL: Repair with json_repair also failed: {e}")
            return []

    if isinstance(data, list):
        # Ensure all items in the list are dicts with the required keys
        valid_items = [
            item for item in data
            if isinstance(item, dict) and "input" in item and "target_output" in item
        ]
        return valid_items
    else:
        print("⚠️ WARNING: Parsed data is not a list.")
        return []

def generate_dataset(
    model_name: str,
    target_count: int = 10000,
    batch_size: int = 25
):
    """
    Generate a dataset using the specified model and save it to a JSONL file.
    """
    output_dir = "RETENTIONENGINE/retentionengine/datasets"
    output_file = os.path.join(output_dir, "qwen2_memory_aligned_dataset.jsonl")  # Updated name and path
    os.makedirs(output_dir, exist_ok=True)  # Create dirs if needed

    print(f"🚀 Loading model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        print(f"✅ Model loaded successfully. Output file: {output_file}")
    except Exception as e:
        print(f"🆘 ERROR: Failed to load model: {e}")
        return []  # Early exit on load failure

    generated_samples = []
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        while len(generated_samples) < target_count:
            remaining = target_count - len(generated_samples)
            current_batch_size = min(batch_size, remaining)
            print(f"🔄 Generating: {len(generated_samples)} / {target_count} | Current batch size: {current_batch_size}")

            prompt = make_seed_prompt(current_batch_size)
            
            # Apply the Qwen2 chat template
            system_prompt = "<|im_start|>system\nYou are a helpful assistant that generates high-quality synthetic data.<|im_end|>"
            user_prompt = f"<|im_start|>user\n{prompt}<|im_end|>"
            assistant_prompt = "<|im_start|>assistant"
            
            final_text = f"{system_prompt}\n{user_prompt}\n{assistant_prompt}"
            
            inputs = tokenizer(final_text, return_tensors="pt").to(model.device)

            # Dynamically calculate max_new_tokens to avoid exceeding context window
            input_len = inputs["input_ids"].shape[1]
            # Set a reasonable generation cap and respect the model's context length (32768 for Qwen2)
            max_new = min(16000, 32768 - input_len - 10)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        temperature=0.5,
                        do_sample=True,
                        top_p=0.9,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id
                    )

                # Decode only the newly generated tokens
                input_token_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0][input_token_length:]
                content = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                new_samples = parse_json_array(content)
                print(f"✅ Extracted {len(new_samples)} valid samples from this batch.")

                if not new_samples:
                    print("⚠️ Failed to generate valid samples in this batch. Retrying after a delay.")
                    time.sleep(5)
                    continue

                for sample in new_samples:
                    if len(generated_samples) >= target_count:
                        break
                    sample["id"] = str(uuid.uuid4())  # Assign a unique ID
                    f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    generated_samples.append(sample)
                
                f_out.flush()  # Write data to disk periodically
                print(f"📊 Total generated: {len(generated_samples)} / {target_count}")

                time.sleep(2)  # Small delay between API calls
            except Exception as e:
                print(f"🆘 ERROR during generation: {e}. Retrying after delay.")
                time.sleep(5)
                continue

    print(f"\n🎉 Generation successfully completed!")
    print(f"📁 Data saved to: {output_file}")
    print(f"📊 Total samples generated: {len(generated_samples)}")
    return generated_samples

if __name__ == "__main__":
    generate_dataset(
        model_name="Qwen/Qwen2-7B-Instruct",  # Fixed to a real model; adjust size if needed (e.g., -1.5B for smaller)
        target_count=5000,
        batch_size=25
    )