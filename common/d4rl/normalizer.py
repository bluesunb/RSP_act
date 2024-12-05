import jax
import numpy as np
from src.common.dataset import Dataset

from typing import Sequence


class TrajNormalizer:
    def __init__(self, dataset: Dataset, keys: Sequence[str] = None, **kwargs):
        self.mean = jax.tree.map(lambda x: np.zeros_like(x), dataset.to_dict())
        self.std = jax.tree.map(lambda x: np.ones_like(x), dataset.to_dict())

        self.keys = keys or []
        for key in self.keys:
            self.mean[key] = np.mean(dataset.get(key), axis=0)
            self.std[key] = np.std(dataset.get(key), axis=0)
            self.std[key][self.std[key] < 1e-6] = 1.0

    def normalize(self, data: Dataset):
        return jax.tree.map(lambda x, m, s: (x - m) / s, data, self.mean, self.std)

    def denormalize(self, data: Dataset):
        return jax.tree.map(lambda x, m, s: x * s + m, data, self.mean, self.std)

    def inverse(self, data: Dataset):
        return self.denormalize(data)

    def __call__(self, data: Dataset):
        return self.normalize(data)
