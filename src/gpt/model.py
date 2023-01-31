""" Transformer model."""
import torch
import torch.nn as nn
from torch.nn import functional as F

from modules import TransformerBlock
from constants import *

class GPT(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()

        self.embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.positional_embedding = nn.Embedding(CONTEXT_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[TransformerBlock(N_EMBED, N_HEADS) for _ in range(N_LAYERS)])
        self.layer_norm = nn.LayerNorm(N_EMBED)
        self.output_layer = nn.Linear(N_EMBED, vocab_size)

    def forward(self, idx, targets=None) -> tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass."""
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

    def save_model(self, save_path: str, state_dict: bool=True):
        if state_dict:
            torch.save(self.state_dict(), save_path)

        else:
            torch.save(self, save_path)
