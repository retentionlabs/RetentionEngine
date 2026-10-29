import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from retentionengine.datasets import LongTextDataset


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
