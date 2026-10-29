import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class LongTextDataset(Dataset):
    """Custom Dataset for formatting and tokenizing long text documents."""

    def __init__(self, documents, tokenizer, max_length=8192):
        self.tokenizer = tokenizer
        self.documents = documents
        self.max_length = max_length

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        # 데이터셋에서 텍스트 문서 하나를 가져옴
        text = self.documents[idx]["text"]

        # 텍스트를 토크나이징 (special token 추가 여부는 토크나이저 설정을 따름)
        tokenized_output = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,  # max_length에 맞춰 자르기
            max_length=self.max_length,
            padding="max_length",  # max_length에 맞춰 패딩
        )

        input_ids = torch.tensor(tokenized_output["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(
            tokenized_output["attention_mask"], dtype=torch.long
        )

        # 언어 모델 학습용 labels: 패딩 위치만 손실 계산에서 제외.
        # pad_token == eos_token인 경우에도 실제 EOS는 attention_mask=1이므로 유지됨.
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def get_pg19_dataloader(tokenizer, batch_size, split="train", max_length=8192):
    """
    Creates a DataLoader for the PG19 dataset.

    Args:
        tokenizer: The tokenizer to use. Needs a pad token (e.g. ``tokenizer.pad_token = tokenizer.eos_token``).
        batch_size: The batch size for the DataLoader.
        split: The dataset split to use ('train', 'validation', or 'test').
        max_length: Tokens per sample (documents are truncated and padded to this length).
    """
    # PG19 데이터셋 로드
    dataset = load_dataset("pg19", split=split)

    # Dataset 객체 생성
    longtext_dataset = LongTextDataset(dataset, tokenizer, max_length=max_length)

    # DataLoader 생성
    dataloader = DataLoader(
        longtext_dataset, batch_size=batch_size, shuffle=(split == "train")
    )

    return dataloader
