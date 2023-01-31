""" Generative Pretrained Transformer.
An implementation of the decoder block of the Original Transformer,
used to create a language model using text from a  given text file.
"""

import tqdm
from typing import Literal

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(117)

from constants import *

print('Model Training Specs:\n')
print(f'Batch Size: {BATCH_SIZE}')
print(f'Context Size: {CONTEXT_SIZE}')
print(f'Training Iterations: {MAX_ITERS}')
print(f'Learning Rate: {LEARNING_RATE}')
print(f'Device: {DEVICE}\n')

with open(FILE_PATH, 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

character_map = { ch:i for i,ch in enumerate(chars)}
index_map = { i:ch for i,ch in enumerate(chars)}

def encode_string(text: str) -> list[int]:
    """ Encodes a string a character level using the character mapping."""
    return [character_map[char] for char in text]

def decode_indexes(tokens: list[int]) -> str:
    """ Decodes a list of integers to their corresponding tokens."""
    return ''.join([index_map[tok] for tok in tokens])


data = torch.tensor(
    encode_string(text=text), dtype=torch.long)

# Store and return tuple if val_split is provided.
if VAL_SPLIT > 0:
    assert VAL_SPLIT < 1
    split_number = int(VAL_SPLIT*len(data))
    training_data = data[:split_number]
    val_data = data[split_number:]

# Store and return single tensor otherwise.
else:
    raise ValueError("Variable 'val_split' must be between 0 and 1.")


def get_batch(split: Literal['train', 'val']) -> tuple[torch.Tensor, torch.Tensor]:
    """ Returns a 'batch_size' number of random samples
    from the training or validation set.
    """
    data = training_data if split == 'train' else val_data

    # Batch size number of random indexes.
    idxs = torch.randint(len(data) - CONTEXT_SIZE, (BATCH_SIZE,))

    X = torch.stack([data[i:i+CONTEXT_SIZE] for i in idxs])
    Y = torch.stack([data[i+1:i+CONTEXT_SIZE+1] for i in idxs])
    X, Y = X.to(DEVICE), Y.to(DEVICE)

    return X, Y

@torch.no_grad()
def estimate_loss() -> dict:
    """ Returns the average performance on training and validation splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = loss.mean()
    model.train()

    return out


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size: int, decoder: bool=True) -> None:
        super().__init__()
        self.is_decoder = decoder
        
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)
        if self.is_decoder:
            self.register_buffer('lower_tri', torch.tril(torch.ones(CONTEXT_SIZE, CONTEXT_SIZE)))

    def forward(self, x: torch.Tensor):
        _,T,C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weights = q @ k.transpose(-2, -1) * (C**-0.5) # Scaled attention.
        if self.is_decoder:
            weights = weights.masked_fill(self.lower_tri[:T,:T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        
        out = weights @ v

        return out


class MultiHeadAttention(nn.Module):
    """ Parallel self attention heads."""
    def __init__(self, n_heads: int, head_size: int) -> None:
        super().__init__()
        self.attention_heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        x = self.dropout(self.projection(x))
        return x


class FeedForwardBlock(nn.Module):
    """ Token-level feed forward network."""
    def __init__(self, n_embed: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed: int, n_heads: int) -> None:
        super().__init__()
        head_size = n_embed // n_heads
        self.atention_block = MultiHeadAttention(n_heads, head_size)
        self.linear_net = FeedForwardBlock(n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor):
        x = x + self.atention_block(self.layer_norm1(x))
        x = x + self.linear_net(self.layer_norm2(x))

        return x


class GPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.positional_embedding = nn.Embedding(CONTEXT_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_EMBED, N_HEADS) for _ in range(N_LAYERS)])
        self.layer_norm = nn.LayerNorm(N_EMBED)
        self.output_layer = nn.Linear(N_EMBED, vocab_size)

    def forward(self, idx, targets=None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        token_embeddings = self.embedding_table(idx)
        pos_embeddings = self.positional_embedding(torch.arange(T, device=DEVICE))

        x = token_embeddings + pos_embeddings
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.output_layer(x)

        if targets is None:
            loss = None
        
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, n_new_tokens: int) -> torch.Tensor:
        """ Generate 'n_new_tokens' number of tokens for the context 'idx'.

        Args:
            idx (torch.Tensor): A (batch_size, T) tensor or array of indices in current context,
                where T is the number of tokens in our input context.
                Note that T cannot exceed the maximum context size seen during training.
            max_new_tokens (int): Number of new tokens to generate.

        Returns:
            torch.Tensor: A (batch_size, T+n_new_tokens) tensor.
        """
        for _ in range(n_new_tokens):
            idx_cropped = idx[:, -CONTEXT_SIZE:] # Crop to maximum context size.
            logits, loss = self(idx_cropped)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # batch_sizex1.
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


m = GPT()
model = m.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for iter in range(MAX_ITERS):
    if iter % EVAL_ITERS == 0:
        losses = estimate_loss()
        print(f"step {iter}, train loss {losses['train']:.4f}, val loss {losses['val']:4f}")

    X_batch, Y_batch = get_batch('train')

    logits, loss = model(X_batch, Y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Using generation.
context = torch.zeros((1, 1), dtype=torch.long).to(DEVICE)
print(decode_indexes(model.generate(context, 100)[0].tolist()))
