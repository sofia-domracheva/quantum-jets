import random
import numpy as np
from numpy.random import Generator

def set_seed(seed: int) -> Generator:
    """Set the random seed and return a NumPy generator."""
    random.seed(seed)
    return np.random.default_rng(seed)  