from numbers import Number
from pathlib import Path
from typing import Any, Dict, Literal, Sequence

import jax
import jax.numpy as jp
import numpy as np
import orbax.checkpoint as ckpt

from src.common.train_state import TrainState
from src.common.utils import func_timeit


def make_rngs(rng: jax.random.PRNGKey, keys: Sequence[str]) -> Dict[str, jax.random.PRNGKey]:
    if len(keys) == 0:
        return jax.random.split(rng)
    rngs = jax.random.split(rng, len(keys) + 1)
    return rng, {k: r for k, r in zip(keys, rngs[1:])}


@func_timeit
def prepare_ckpt(ckpt_dir: Path, monitor: str, best_mode: Literal["max"], keep_n: int):
    options = ckpt.CheckpointManagerOptions(
        max_to_keep=keep_n,
        best_fn=lambda metrics: metrics[monitor],
        best_mode=best_mode,
        cleanup_tmp_directories=True,
        create=True,
    )

    item_handlers = {
        "params": ckpt.StandardCheckpointHandler(),
        "extra_variables": ckpt.StandardCheckpointHandler(),
    }

    manager = ckpt.CheckpointManager(
        directory=Path(ckpt_dir).absolute(),
        item_names=tuple(item_handlers.keys()),
        item_handlers=item_handlers,
        options=options,
    )

    return manager


def save_model(manager: ckpt.CheckpointManager, state: TrainState, metrics: Dict[str, Any], epoch_idx: int):
    save_items = {"params": ckpt.args.StandardSave(state.params)}
    if state.extra_variables:
        save_items["extra_variables"] = ckpt.args.StandardSave(state.extra_variables)
    save_items = ckpt.args.Composite(**save_items)
    metrics = {k: v for k, v in metrics.items() if isinstance(v, (Number, str, bool))}
    manager.save(step=epoch_idx, args=save_items, metrics=metrics)


def load_model(manager: ckpt.CheckpointManager, epoch_idx: int = -1) -> Dict[str, Any]:
    if epoch_idx < 0:
        epoch_idx = manager.best_step()

    state_dict = manager.restore(epoch_idx)
    state_dict = {k: v for k, v in state_dict.items() if v is not None}
    return state_dict


def batch_pad(fn, batch_size: int, batch_argnum: int = 0, unpad_out_argnums=(0,)):
    """
    Pad the batch dimension of the batch input if it has smaller size than the regular batch size.

    Args:
        fn:                 Function to be padded.
        batch_size:         Regular batch size.
        batch_argnum:       Positional index of the batch argument.
        unpad_out_argnums:  Positional indices of the output arguments to be unpadded.

    Returns:
        Padded function.
    """

    def get_pad_width(x, n_pad):
        pad_width = [(0, n_pad)] + [(0, 0)] * (x.ndim - 1)
        return pad_width

    def pad_arg(x, n_pad, pad_fn):
        return jax.tree.map(lambda x: pad_fn(x, get_pad_width(x, n_pad), mode="constant", constant_values=0), x)

    def unpad_arg(x, n_pad):
        return jax.tree.map(lambda x: x[:-n_pad] if isinstance(x, jax.typing.ArrayLike) else x, x)

    def padded_fn(*args, **kwargs):
        # n = len(args[batch_argnum])
        n = jax.tree.leaves(jax.tree.map(lambda x: x.shape[0], args[batch_argnum]))
        types = jax.tree.leaves(jax.tree.map(type, args[batch_argnum]))
        assert len(set(n)) == 1, "Batch size should be the same for all inputs"
        assert len(set(types)) == 1, "All batch inputs should have the same type"
        n = n[0]
        batch_type = types[0]
        pad_fn = np.pad if batch_type == np.ndarray else jp.pad

        n_pad = (batch_size - n % batch_size) % batch_size
        if n_pad > 0:
            args = list(args)
            args[batch_argnum] = pad_arg(args[batch_argnum], n_pad, pad_fn)

        out = fn(*args, **kwargs)

        if n_pad > 0:
            if isinstance(out, tuple):
                out = list(out)
                for i in unpad_out_argnums:
                    out[i] = unpad_arg(out[i], n_pad)
                out = tuple(out)
            elif len(unpad_out_argnums) == 1:
                out = unpad_arg(out, n_pad)

        return out

    return padded_fn
