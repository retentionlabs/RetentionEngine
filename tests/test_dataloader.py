import importlib
import os

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from retentionengine.datasets import LongTextDataset, get_pg19_dataloader
from retentionengine.datasets.dataloader import PG19_DATASET, PG19_REVISION


def _tokenizer_with_pad_eq_eos():
    vocab = {"[UNK]": 0, "<eos>": 1, "hello": 2, "world": 3}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", eos_token="<eos>", pad_token="<eos>"
    )


def test_padding_masked_but_real_eos_kept():
    tokenizer = _tokenizer_with_pad_eq_eos()
    dataset = LongTextDataset([{"text": "hello world <eos>"}], tokenizer, max_length=6)
    item = dataset[0]

    assert item["input_ids"].tolist() == [2, 3, 1, 1, 1, 1]
    assert item["attention_mask"].tolist() == [1, 1, 1, 0, 0, 0]
    # The real EOS (position 2) stays a target; only padding is ignored.
    assert item["labels"].tolist() == [2, 3, 1, -100, -100, -100]
    assert item["labels"].dtype == torch.long


def test_pg19_loader_uses_pinned_parquet_copy(monkeypatch):
    calls = []

    def fake_load_dataset(path, **kwargs):
        calls.append((path, kwargs))
        return [{"text": "hello world"}, {"text": "world <eos>"}]

    dataloader_module = importlib.import_module("retentionengine.datasets.dataloader")
    monkeypatch.setattr(dataloader_module, "load_dataset", fake_load_dataset)
    loader = get_pg19_dataloader(
        _tokenizer_with_pad_eq_eos(), batch_size=2, split="validation", max_length=4
    )

    # Not the script-only "pg19" repo, which datasets>=4 refuses to load.
    assert calls == [
        ("emozilla/pg19", {"split": "validation", "revision": PG19_REVISION})
    ]
    batch = next(iter(loader))
    assert batch["input_ids"].tolist() == [[2, 3, 1, 1], [3, 1, 1, 1]]
    assert batch["labels"].tolist() == [[2, 3, -100, -100], [3, 1, -100, -100]]


@pytest.mark.skipif(
    os.environ.get("HF_HUB_OFFLINE") != "0",
    reason="needs the Hugging Face Hub; run with HF_HUB_OFFLINE=0",
)
def test_pg19_parquet_copy_resolves_on_installed_datasets():
    # Metadata only (README/parquet listing), no shard download.
    from datasets import load_dataset_builder

    builder = load_dataset_builder(PG19_DATASET, revision=PG19_REVISION)
    assert "text" in builder.info.features
    assert {name: s.num_examples for name, s in builder.info.splits.items()} == {
        "train": 28602,
        "validation": 50,
        "test": 100,
    }
