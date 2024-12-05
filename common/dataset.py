import re
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Tuple, Type, TypeVar, Union

import jax
import numpy as np
from flax.struct import dataclass
from flax.traverse_util import flatten_dict, unflatten_dict

T = TypeVar("T")


def from_dict(data_class: Type[T], data: Dict[str, Any]) -> T:
    if not hasattr(data_class, "__dataclass_fields__"):
        # if not a dataclass, return as is
        return data
    field_types = {f.name: f.type for f in data_class.__dataclass_fields__.values()}
    return data_class.create(**{k: from_dict(field_types[k], v) for k, v in data.items()})


def _indent(x: str, num_spaces: int = 4) -> str:
    indent_str = " " * num_spaces
    lines = x.split("\n")
    assert not lines[-1], "Last line should be empty"
    # skip the final line because it's empty and should not be indented
    return "\n".join([indent_str + line for line in lines[:-1]]) + "\n"


def stack_dict(data_dict: dict):
    def _stack_dict(data):
        if isinstance(data, (list, tuple)):
            if isinstance(data[0], dict):
                # list of dicts format
                data = [flatten_dict(d, sep="/") for d in data]
                new_data = {k: np.stack([d[k] for d in data]) for k in data[0].keys()}
                new_data = unflatten_dict(new_data, sep="/")
            else:
                # list of array-like format
                new_data = np.stack(data)
        else:
            # already a stacked dict
            new_data = data
        return new_data

    return {k: _stack_dict(v) for k, v in data_dict.items()}


def spec(data):
    """
    Generate a structure dictionary representing the shape and dtype of the given data.
    """
    if is_dataclass(data):
        data = asdict(data)

    structure = {}
    if isinstance(data, dict):
        for k, v in data.items():
            structure[k] = spec(v)
    elif isinstance(data, (list, tuple)):
        structure = spec(data[0])
    elif hasattr(data, "shape"):
        structure = {"shape": data.shape, "dtype": data.dtype}
    else:
        structure = {"shape": (), "dtype": type(data)}
    return structure


@dataclass
class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, ids: Union[int, slice]) -> "Dataset":
        return jax.tree.map(lambda x: x[ids], self)

    @classmethod
    def create(cls, **kwargs):
        instance = cls(**kwargs)
        instance.freeze()
        return instance

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return from_dict(cls, data)

    @classmethod
    def from_tuple(cls, data: Tuple[Any]):
        kwargs = {f.name: data[i] for i, f in enumerate(cls.__dataclass_fields__.values())}
        return cls.create(**kwargs)

    def to_dict(self):
        return asdict(self)

    def to_tuple(self):
        return tuple(asdict(self).values())

    def freeze(self):
        return jax.tree.map(lambda x: x.setflags(write=False), self)

    def unfreeze(self):
        return jax.tree.map(lambda x: x.setflags(write=True), self)

    def atleast_3d(self):
        return jax.tree.map(lambda x: np.atleast_3d(x), self)

    def sample(self, batch_size: int = 1, seed: int = None):
        generator = np.random.default_rng(seed)
        ids = generator.integers(0, len(self), size=(batch_size,))
        return self[ids]

    def pretty_repr(self, num_spaces=4):
        """Returns an indented representation of the nested dictionary."""

        def pretty_dict(x):
            if not isinstance(x, dict):
                return repr(x)
            rep = ""
            for key, val in x.items():
                rep += f"{key}: {pretty_dict(val)}\n"
            if rep:
                return "\n" + _indent(rep, num_spaces)
            else:
                return "{}"

        s = f"{self.__class__.__name__}({pretty_dict(asdict(jax.tree.map(np.shape, self)))})"
        s = re.sub(r"\s+\n+", "\n", s)
        return s

    def __str__(self):
        return self.pretty_repr()
