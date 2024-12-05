from pathlib import Path
from typing import Sequence, Callable

import tensorflow as tf


def tf_cross_file_batch_loader(
    files: Sequence[str | Path],
    sample_fn: Callable[[str], tf.Tensor],
    batch_size: int,
    drop_last: bool = False,
    shuffle: bool = False,
    prefetch_buffer: int = tf.data.experimental.AUTOTUNE,
    num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
) -> tf.data.Dataset:
    """
    Load files from a list of paths and sample a batch over these files.
    """

    assert len(files) > 0, "No files to load"
    if isinstance(files[0], Path):
        files = list(map(str, files))

    dataset = tf.data.Dataset.from_tensor_slices(files)

    if shuffle:
        dataset = dataset.shuffle(len(files))

    dataset = dataset.map(sample_fn, num_parallel_calls=num_parallel_calls)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=drop_last)
    dataset = dataset.prefetch(prefetch_buffer)
    dataset = dataset.as_numpy_iterator()
    return dataset
