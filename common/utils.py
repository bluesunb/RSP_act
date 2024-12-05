import time
from functools import wraps
from numbers import Number
from pathlib import Path
from typing import Dict, Union

import flax.linen as nn
import jax
import jax.numpy as jp
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import numpy as np
from flax.jax_utils import unreplicate
from flax.traverse_util import flatten_dict
from tabulate import tabulate

import src.common.logger as logger

Array = Union[np.ndarray, jax.Array]


def func_timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"[TimeIt] {func.__name__} ... ", end="")
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug(f"took {end - start:.3f} sec")
        return result

    return wrapper


def reduce_array_to_scalar(array: Array) -> Number | Array:
    """Reduce an array to scalar value if the array has only one element."""
    if isinstance(array, Array) and array.size == 1:
        return array.item()
    return array


def maybe_reduce(pytree):
    pytee = reduce_array_to_scalar(unreplicate(pytree))
    return jax.tree.map(reduce_array_to_scalar, pytee)


def tabulate_model(rngs, model: nn.Module, **sample_input: jax.Array) -> str:
    return model.tabulate(
        rngs,
        **sample_input,
        train=True,
        console_kwargs={"force_terminal": False, "width": 300},
    )


def tabulate_param(params) -> str:
    def readable_size(nbytes):
        for unit in ["B", "KB", "MB", "GB"]:
            if nbytes < 1024:
                return f"{nbytes:.2f} {unit}"
            nbytes /= 1024
        return f"{nbytes:.2f} TB"

    params = flatten_dict(params, sep=".")
    summary = {
        "Name": list(params.keys()),
        "Shape": list(jax.tree.map(jp.shape, params).values()),
        "Count": list(map(np.prod, jax.tree.map(jp.shape, params).values())),
        "Mem": list(map(lambda x: readable_size(x.nbytes), jax.tree.map(jp.asarray, params).values())),
    }
    table = tabulate(summary, headers="keys", tablefmt="pretty")
    total_mem = sum(jax.tree.leaves(jax.tree.map(lambda x: x.nbytes, params)))
    table += f"\nTotal memory: {readable_size(total_mem)}"
    return table


def tabulate_info(params: Dict[str, jp.ndarray], log_dir: Path) -> str:
    params_summary = tabulate_param(params)
    with open(log_dir / "params_summary.txt", "w") as f:
        f.write(params_summary)
    logger.info(f"Summary saved to {log_dir}")


def plot_to_img(plot: plt.Figure):
    canvas = agg.FigureCanvasAgg(plot)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    w, h = plot.canvas.get_width_height()
    img = img.reshape((h, w, -1))
    plt.close(plot)
    return img