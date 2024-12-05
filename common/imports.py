import importlib
import inspect
import sys
from typing import Any

from src.common.aliases import ALIASE


def resolve_import(import_path: str | Any, raise_exception=True) -> Any:
    """Resolves on object from a string import path, or returns the input without touch."""
    if isinstance(import_path, str):
        import_path = resolve_import_from_str(import_path, raise_exception=raise_exception)
    return import_path


def resolve_import_from_str(import_str: str, raise_exception=True) -> Any:
    """Resolves an object from a string import path."""
    if import_str in ALIASE:
        import_str = ALIASE[import_str]

    if "." in import_str:
        module_path, class_name = import_str.rsplit(".", 1)
        module = importlib.import_module(module_path)
        resolved_class = getattr(module, class_name, None)
    else:
        resolved_class = getattr(sys.modules[__name__], import_str, None)

    if resolved_class is None and raise_exception:
        raise ImportError(f"Could not import {import_str}")

    return resolved_class


def class_to_name(x: Any) -> str | Any:
    """
    Returns the name of a class as a string.

    Args:
        x: Class to get the name of.

    Returns:
        (str | Any): Name of the class.

    Examples:
        >>> class_to_name(int)
        'int'
        >>> class_to_name(1)
        1
        >>> class_to_name(Module)
        'torch.nn.modules.module.Module'
    """
    if inspect.isclass(x):
        return f"{inspect.getmodule(x).__name__}.{x.__name__}"
    return x
