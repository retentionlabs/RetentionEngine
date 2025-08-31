# -*- coding: utf-8 -*-
"""
Qwen3 기반 메모리 모듈 학습용 데이터셋 생성기 (안정화 버전)
------------------------------------------------
생성된 데이터: RETENTIONENGINE/retentionengine/datasets/qwen3_memory_aligned_dataset.jsonl
→ 2/3 한국어, 1/3 영어 샘플 포함
"""

import os
import json
import re
import time
import random
import uuid
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json_repair


def make_seed_prompt(batch_size: int = 100) -> str:
    """
    2/3 확률로 한국어, 1/3 확률로 영어 프롬프트 반환
    """
    if random.random() < 0.33:  # 약 1/3 확률로 영어
        return make_seed_prompt_english(batch_size)
    else:
        return make_seed_prompt_korean(batch_size)


def make_seed_prompt_korean(batch_size: int = 5) -> str:
    """
    한국어 프롬프트
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
  "target_output": "Qwen3이 생성할 것으로 예상되는 출력",
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
- 오직 하나의 완성된 JSON 배열만 출력해줘. 설명, 마크다운, ```json``` 태그는 절대 포함하지 마.
- 예시: `[ {{...}}, {{...}}, ... ]`
"""
    return prompt.strip()


def make_seed_prompt_english(batch_size: int = 100) -> str:
    """
    영어 프롬프트 (1/3 확률로 호출)
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
  "target_output": "Output expected from Qwen3",
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
- Example: `[ {{...}}, {{...}}, ... ]`
"""
    return prompt.strip()


# [수정] JSON 배열을 안정적으로 파싱하는 함수
def parse_json_array(text: str) -> List[Dict]:
    """
    생성된 텍스트에서 유효한 JSON 배열을 추출
    """
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        print("⚠️ 경고: 응답에서 JSON 배열을 찾지 못했습니다.")
        return []

    json_array_str = match.group(0)
    try:
        data = json.loads(json_array_str)
    except json.JSONDecodeError:
        data = json_repair.loads(json_array_str)

    if isinstance(data, list):
        valid_items = [
            item for item in data
            if isinstance(item, dict) and "input" in item and "target_output" in item
        ]
        return valid_items
    else:
        print("⚠️ 경고: 파싱된 데이터가 리스트가 아닙니다.")
        return []


def generate_dataset(
    target_count: int = 10000,
    batch_size: int = 100
):
    """
    Qwen3을 사용해 데이터셋을 생성하고, 지정된 경로에 저장
    """
    output_dir = os.path.join("retentionengine", "datasets")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "qwen3_memory_aligned_dataset.jsonl")

    print("🚀 모델을 로딩 중입니다... (Qwen/Qwen2-7B-Instruct)")
    model_name = "Qwen/Qwen2-7B-Instruct" 

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    print(f"✅ 모델 로딩 완료. 저장 위치: {os.path.abspath(output_file)}")

    generated_samples = []
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        while len(generated_samples) < target_count:
            remaining = target_count - len(generated_samples)
            current_batch_size = min(batch_size, remaining)
            print(f"🔄 생성 중: {len(generated_samples)} / {target_count} | 이번 배치: {current_batch_size}")

            prompt = make_seed_prompt(current_batch_size)
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that generates high-quality synthetic data."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            # 안전한 max_new_tokens 계산
            input_len = inputs["input_ids"].shape[1]
            max_new = min(16000, model.config.max_position_embeddings - input_len)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )

            input_token_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_token_length:]
            content = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            new_samples = parse_json_array(content)
            print(f"✅ 이번 배치에서 {len(new_samples)}개의 유효한 샘플 추출")

            if not new_samples:
                print("⚠️ 이번 배치에서 유효한 샘플을 생성하지 못했습니다. 재시도합니다.")
                time.sleep(5)
                continue

            for sample in new_samples:
                if len(generated_samples) >= target_count:
                    break
                sample["id"] = str(uuid.uuid4())  # uuid 기반 ID
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                generated_samples.append(sample)
            
            f_out.flush()
            print(f"📊 누적 생성: {len(generated_samples)} / {target_count}")

            time.sleep(2)

    print(f"\n🎉 성공적으로 완료!")
    print(f"📁 저장 위치: {os.path.abspath(output_file)}")
    print(f"📊 총 생성된 샘플 수: {len(generated_samples)}")
    return generated_samples


if __name__ == "__main__":
    samples = generate_dataset(
        target_count=5000,
        batch_size=25
    )
