from pathlib import Path
from typing import Sequence, Type

import numpy as np

from src.common.dataset import Dataset


class NPDataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = False,
        transform_fn: callable = None,
        seed: int = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.transform_fn = transform_fn if transform_fn is not None else lambda x: x
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        n_batches, remainder = divmod(len(self.dataset), self.batch_size)
        return n_batches + (remainder > 0 and not self.drop_last)

    def __getitem__(self, ids):
        batch = self.dataset[ids]
        batch = self.transform_fn(batch)
        return batch

    def __iter__(self):
        ids = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.permutation(ids)

        for i in range(len(self)):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            yield self.__getitem__(ids[s])


class SequentialFileLoader:
    def __init__(
        self,
        files: Sequence[Path],
        load_fn: callable,
        loader_def: Type[Dataset] = Dataset,
        shuffle_files: bool = False,
        **loader_kwargs,
    ):
        self.files = files
        self.load_fn = load_fn
        self.loader_def = loader_def
        self.shuffle_files = shuffle_files
        self.loader_kwargs = loader_kwargs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        dataset = self.load_fn(self.files[idx])
        return self.loader_def(dataset, **self.loader_kwargs)

    def __iter__(self):
        ids = np.arange(len(self))
        if self.shuffle_files:
            np.random.shuffle(ids)

        for i in ids:
            yield self.__getitem__(i)


class BatchedFileLoader:
    def __init__(
        self,
        files: Sequence[Path],
        load_fn: callable,
        loader_def: Type[Dataset] = Dataset,
        batch_size_file: int = 32,
        drop_last_file: bool = False,
        shuffle_files: bool = False,
        **loader_kwargs,
    ):
        self.files = files
        self.load_fn = load_fn
        self.loader_def = loader_def
        self.batch_size_file = batch_size_file
        self.drop_last_file = drop_last_file
        self.shuffle_files = shuffle_files
        self.loader_kwargs = loader_kwargs
        
    def __len__(self):
        n_batches, remainder = divmod(len(self.files), self.batch_size_file)
        return n_batches + (remainder > 0 and not self.drop_last_file)
    
    def __getitem__(self, ids):
        