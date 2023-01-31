import torch
import torch.nn as nn
""" Individual modules for constructing GPT."""
from torch.nn import functional as F

from constants import *

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


class FeedForwardNet(nn.Module):
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


class TransformerBlock(nn.Module):
    def __init__(self, n_embed: int, n_heads: int) -> None:
        super().__init__()
        head_size = n_embed // n_heads
        self.atention_block = MultiHeadAttention(n_heads, head_size)
        self.linear_net = FeedForwardNet(n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor):
        x = x + self.atention_block(self.layer_norm1(x))
        x = x + self.linear_net(self.layer_norm2(x))

        return x
