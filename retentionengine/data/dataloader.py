# data/dataloader.py

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import defaultdict


def _extract_conversations(dataset):
    """Extracts conversation threads from the oasst1 dataset."""
    message_map = {msg['message_id']: msg for msg in dataset}

    threads = defaultdict(list)
    for msg in dataset:
        threads[msg['parent_id']].append(msg)

    conversations = []
    for root in threads.get(None, []):
        # Build a thread from the root message
        thread = []
        queue = [root]
        visited = set()

        while queue:
            message = queue.pop(0)
            if message['message_id'] in visited:
                continue
            visited.add(message['message_id'])

            thread.append({
                "role": message['role'],
                "text": message['text'],
                "rank": message['rank']
            })

            children = sorted(threads.get(message['message_id'], []), key=lambda x: x['rank'])
            queue.extend(children)

        # Only include threads with at least one assistant response
        if any(msg['role'] == 'assistant' for msg in thread):
            conversations.append(thread)

    return conversations


class DialogueDataset(Dataset):
    """Custom Dataset for formatting and tokenizing dialogue conversations."""

    def __init__(self, conversations, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.conversations = conversations
        self.max_length = max_length
        self.role_tokens = {"prompter": "<|prompter|>", "assistant": "<|assistant|>"}

        # Add special role tokens to the tokenizer if they don't exist
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(self.role_tokens.values())}
        )

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]

        input_ids = []
        labels = []

        # Start with the beginning-of-sequence token
        input_ids.append(self.tokenizer.bos_token_id)
        labels.append(-100)  # Do not compute loss on the BOS token

        for utterance in conversation:
            role_token = self.role_tokens[utterance['role']]
            message_tokens = self.tokenizer(
                f"{role_token}{utterance['text']}{self.tokenizer.eos_token}",
                add_special_tokens=False
            ).input_ids

            input_ids.extend(message_tokens)

            if utterance['role'] == 'prompter':
                # Mask out prompter's messages and role token in labels
                labels.extend([-100] * len(message_tokens))
            else:  # assistant
                # Only compute loss on the assistant's messages
                labels.extend(message_tokens)

        # Truncate to max_length
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]

        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
        labels.extend([-100] * padding_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def get_oasst_dataloader(tokenizer, batch_size, split='train', lang=None):
    """
    Creates a DataLoader for the OpenAssistant/oasst1 dataset.

    Args:
        tokenizer: The tokenizer to use.
        batch_size: The batch size for the DataLoader.
        split: The dataset split to use ('train' or 'validation').
        lang: The language code to filter by (e.g., 'en', 'ko').
              If None, all languages are used.
    """
    dataset = load_dataset("OpenAssistant/oasst1", split=split)

    if lang:
        dataset = dataset.filter(lambda example: example['lang'] == lang)

    # Filter for high-quality, ready-to-use data
    dataset = dataset.filter(lambda x: x["rank"] is not None and x["rank"] == 0)

    conversations = _extract_conversations(dataset)
    dialogue_dataset = DialogueDataset(conversations, tokenizer)

    dataloader = DataLoader(
        dialogue_dataset,
        batch_size=batch_size,
        shuffle=(split == 'train')
    )

    return dataloader