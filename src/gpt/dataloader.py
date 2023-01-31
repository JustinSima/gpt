""" Class for parsing text file, returning batches, and encoding/decoding sequences."""
import torch
from typing import Literal

from constants import *


class DataLoader:
    def __init__(self) -> None:
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            text = f.read()

        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        self.character_map = { ch:i for i,ch in enumerate(self.chars)}
        self.index_map = { i:ch for i,ch in enumerate(self.chars)}

        data = torch.tensor(
            self.encode_string(text=text), dtype=torch.long)

        # Create train / test splits.
        if VAL_SPLIT >= 0:
            assert VAL_SPLIT < 1
            split_number = int(VAL_SPLIT*len(data))
            self.training_data = data[:split_number]
            self.val_data = data[split_number:]


    def encode_string(self, text: str) -> list[int]:
        """ Encodes a string a character level using the character mapping."""
        return [self.character_map[char] for char in text]

    def decode_indexes(self, tokens: list[int]) -> str:
        """ Decodes a list of integers to their corresponding tokens."""
        return ''.join([self.index_map[tok] for tok in tokens])

    def get_batch(self, split: Literal['train', 'val']) -> tuple[torch.Tensor, torch.Tensor]:
        """ Returns a 'batch_size' number of random samples
        from the training or validation set.
        """
        data = self.training_data if split == 'train' else self.val_data

        # Batch size number of random indexes.
        idxs = torch.randint(len(data) - CONTEXT_SIZE, (BATCH_SIZE,))

        x = torch.stack([data[i:i+CONTEXT_SIZE] for i in idxs])
        y = torch.stack([data[i+1:i+CONTEXT_SIZE+1] for i in idxs])
        x, y = x.to(DEVICE), y.to(DEVICE)

        return x, y
