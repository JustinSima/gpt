""" Create, train, and save a GPT-2 inspired transformer model on text file specified in 'constants.py'."""
import torch

from dataloader import DataLoader
from model import GPT
from training import train_model
from constants import *


def main():
    """ Train and save a GPT model."""
    data = DataLoader()
    vocab_size = data.vocab_size
    model = GPT(vocab_size=vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_model(data, model, optimizer)

    model.save_model(save_path='gpt-pretrained.pt', state_dict=True)

    # Using generation to create example text.
    context = torch.zeros((1, 1), dtype=torch.long).to(DEVICE)
    print(data.decode_indexes(model.generate(context, 100)[0].tolist()))

if __name__ == '__main__':
    main()
