import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class LongTextDataset(Dataset):
    """Custom Dataset for formatting and tokenizing long text documents."""

    def __init__(self, documents, tokenizer, max_length=8192):  # Gemma 모델에 맞춰 max_length 조절
        self.tokenizer = tokenizer
        self.documents = documents
        self.max_length = max_length

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        # PG19 데이터셋에서 텍스트 문서 하나를 가져옴
        text = self.documents[idx]['text']

        # 텍스트를 토크나이징
        tokenized_output = self.tokenizer(
            text,
            add_special_tokens=True,  # 문장의 시작(BOS)과 끝(EOS) 토큰 추가
            truncation=True,  # max_length에 맞춰 자르기
            max_length=self.max_length,
            padding='max_length'  # max_length에 맞춰 패딩
        )

        input_ids = tokenized_output['input_ids']

        # 언어 모델 학습을 위해 labels를 input_ids를 복사하여 생성
        # 일반적으로 패딩 토큰은 손실 계산에서 제외
        labels = torch.tensor(input_ids)
        labels[labels == self.tokenizer.pad_token_id] = -100  # 손실 계산에서 패딩 부분은 무시

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": labels.to(torch.long)
        }


def get_pg19_dataloader(tokenizer, batch_size, split='train'):
    """
    Creates a DataLoader for the PG19 dataset.

    Args:
        tokenizer: The tokenizer to use.
        batch_size: The batch size for the DataLoader.
        split: The dataset split to use ('train', 'validation', or 'test').
    """
    # PG19 데이터셋 로드
    dataset = load_dataset("pg19", split=split)

    # Dataset 객체 생성
    longtext_dataset = LongTextDataset(dataset, tokenizer)

    # DataLoader 생성
    dataloader = DataLoader(
        longtext_dataset,
        batch_size=batch_size,
        shuffle=(split == 'train')
    )

    return dataloader