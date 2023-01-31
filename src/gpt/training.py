""" Training loop."""
import torch
import torch.nn as nn

from dataloader import DataLoader
from constants import *


@torch.no_grad()
def estimate_loss(data: DataLoader, model: nn.Module) -> dict:
    """ Returns the average performance on training and validation splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = data.get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = loss.mean()
    model.train()

    return out

def train_model(data: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer):
    for iter in range(MAX_ITERS):
        if iter % EVAL_ITERS == 0:
            losses = estimate_loss(data, model)
            print(f"step {iter}, train loss {losses['train']:.4f}, val loss {losses['val']:4f}")

        X_batch, Y_batch = data.get_batch('train')

        _, loss = model(X_batch, Y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
