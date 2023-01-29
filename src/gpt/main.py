from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(117)


class TextFileDataLoader:
    def __init__(self, 
        file_path: str, batch_size: int=16, context_size: int=8
    ):

        # Read data.
        with open(file_path, 'r') as f:
            self.text = f.read()

        self.batch_size = batch_size
        self.context_size = context_size
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        self.character_map = { ch:i for i,ch in enumerate(self.chars)}
        self.index_map = { i:ch for i,ch in enumerate(self.chars)}
        self.training_data = None
        self.val_data = None

        self.load_datasets()

    def encode_string(self, text: str):
        """ Encodes a string a character level using the character mapping."""
        return [self.character_map[char] for char in text]

    def decode_indexes(self, tokens: list[int]):
        """ Decodes a list of integers to their corresponding tokens."""
        return ''.join([self.index_map[tok] for tok in tokens])

    def load_datasets(self, val_split: int=0.1):
        """ Loads text encoded text file as a pytorch tensor.
        If 'val_split' > 0, returns a training and validation tensor.
        """
        data = torch.tensor(
            self.encode_string(text=self.text), dtype=torch.long)

        # Store and return tuple if val_split is provided.
        if val_split > 0:
            assert val_split < 1
            split_number = int(val_split*len(data))
            training_data = train = data[:split_number]
            val_data = data[split_number:]
            
            self.training_data = training_data
            self.val_data = val_data

            return training_data, val_data

        # Store and return single tensor otherwise.
        else:

            self.training_data = data

            return data

    def get_batch(self, split: Literal['train', 'val']):
        data = self.training_data if split == 'train' else self.val_data

        # Batch size number of random indexes.
        idxs = torch.randint(len(data) - self.context_size, (self.batch_size,))

        X = torch.stack([data[i:i+self.context_size] for i in idxs])
        Y = torch.stack([data[i+1:i+self.context_size+1] for i in idxs])

        return X, Y


class Bigram(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        self.embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.embedding_table(idx)

        if targets is None:
            loss = None
        
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, n_new_tokens: int):
        """_summary_

        Args:
            idx : A (batch_size, T) array of indices in current context,
            where T <= max context seen during training.
            max_new_tokens (int): Number of new tokens to generate.

        Returns:
            _type_: _description_
        """
        # idx is (B, T) array of indices in current context.
        for _ in range(n_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # batch_sizex1.
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# # Using generation.
# idx = torch.zeros((1, 1), dtype=torch.long)
# print(dl.decode_indexes(model.generate(idx, 100)[0].tolist()))


file_path = '../../data/tiny-shakespeare.txt'
batch_size = 32

dl = TextFileDataLoader(file_path=file_path, batch_size=batch_size)
model = Bigram(vocab_size=65)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for step in range(20_000):
    X, Y = dl.get_batch('train')

    logits, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

