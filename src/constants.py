import numpy as np

EMBEDDING_DIM = 10
SAMPLE_DIM = 3
LATENT_DIM = 128
SEQ_START = np.array([1] + [0] * (EMBEDDING_DIM - 1))
SEQ_END = np.array([0] + [1] + [0] * (EMBEDDING_DIM - 2))
MARK = np.array([0, 0, 1] + [0] * (EMBEDDING_DIM - 3))
