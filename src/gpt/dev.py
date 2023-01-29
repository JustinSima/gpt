from typing import Literal

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(117)


file_path = '../../data/tiny-shakespeare.txt'
val_split = 0.1
batch_size = 32
context_size = 8

max_iters = 1_000
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # else 'mps' if torch.backends.mps.is_available() \
    # else 'cpu'
eval_iters = 200




with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

character_map = { ch:i for i,ch in enumerate(chars)}
index_map = { i:ch for i,ch in enumerate(chars)}

def encode_string(text: str):
    """ Encodes a string a character level using the character mapping."""
    return [character_map[char] for char in text]

def decode_indexes(tokens: list[int]):
    """ Decodes a list of integers to their corresponding tokens."""
    return ''.join([index_map[tok] for tok in tokens])


data = torch.tensor(
    encode_string(text=text), dtype=torch.long)

# Store and return tuple if val_split is provided.
if val_split > 0:
    assert val_split < 1
    split_number = int(val_split*len(data))
    training_data = train = data[:split_number]
    val_data = data[split_number:]

# Store and return single tensor otherwise.
else:
    training_data = data


def get_batch(split: Literal['train', 'val']):
    data = training_data if split == 'train' else val_data

    # Batch size number of random indexes.
    idxs = torch.randint(len(data) - context_size, (batch_size,))

    X = torch.stack([data[i:i+context_size] for i in idxs])
    Y = torch.stack([data[i+1:i+context_size+1] for i in idxs])
    X, Y = X.to(device), Y.to(device)

    return X, Y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = loss.mean()
    model.train()

    return out

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

m = Bigram(vocab_size=65)
model = m.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {iter}, train loss {losses['train']:.4f}, val loss {losses['val']:4f}")

    X, Y = get_batch('train')

    logits, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Using generation.
context = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode_indexes(model.generate(context, 100)[0].tolist()))
