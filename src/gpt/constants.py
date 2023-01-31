""" Constants for defining and training model."""


FILE_PATH = '../../data/tiny-shakespeare.txt'
VAL_SPLIT = 0.1
BATCH_SIZE = 64
CONTEXT_SIZE = 256

MAX_ITERS = 1_000
LEARNING_RATE = 3e-4
DEVICE = 'cuda'
EVAL_ITERS = 200
N_EMBED = 384
N_LAYERS = 6
N_HEADS = 6
DROPOUT = 0.2
