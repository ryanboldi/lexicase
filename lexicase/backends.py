"""
Backend selection for lexicase selection.

This module never imports jax or torch at load time. A JAX array or a Torch
tensor cannot exist unless the caller has already imported that library, so
`sys.modules` is enough to detect one. Importing `lexicase` therefore never
pulls in an optional backend, even when it is installed.
"""

from __future__ import annotations

import sys
from typing import Any

NUMPY = "numpy"
JAX = "jax"
TORCH = "torch"
AUTO = "auto"

BACKENDS = (NUMPY, JAX, TORCH)

_INSTALL_HINTS = {
    JAX: "The JAX backend requires jax. Install it with: pip install 'lexicase[jax]'",
    TORCH: "The Torch backend requires torch. Install it with: pip install 'lexicase[torch]'",
}


def is_jax_array(x: Any) -> bool:
    """Return True if x is a JAX array, without importing jax."""
    jax = sys.modules.get("jax")
    array_type = getattr(jax, "Array", None)
    if array_type is None:
        return False
    return isinstance(x, array_type)


def is_torch_tensor(x: Any) -> bool:
    """Return True if x is a Torch tensor, without importing torch."""
    torch = sys.modules.get("torch")
    tensor_type = getattr(torch, "Tensor", None)
    if tensor_type is None:
        return False
    return isinstance(x, tensor_type)


def jax_is_available() -> bool:
    """Return True if jax can be imported. Imports jax as a side effect."""
    try:
        import jax  # noqa: F401
    except ImportError:
        return False
    return True


def torch_is_available() -> bool:
    """Return True if torch can be imported. Imports torch as a side effect."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(backend: str, fitness_matrix: Any) -> str:
    """Resolve the requested backend to "numpy", "jax", or "torch".

    Args:
        backend: "auto", "numpy", "jax", or "torch"
        fitness_matrix: The input array, used when backend is "auto"

    Returns:
        One of "numpy", "jax", "torch"
    """
    if backend == AUTO:
        if is_jax_array(fitness_matrix):
            return JAX
        if is_torch_tensor(fitness_matrix):
            return TORCH
        return NUMPY
    if backend in BACKENDS:
        return backend
    raise ValueError(
        f"Unknown backend {backend!r}, expected 'auto', 'numpy', 'jax', or 'torch'"
    )


def load_jax_impl():
    """Import and return lexicase.jax_impl, with a helpful error if jax is missing."""
    try:
        from . import jax_impl
    except ImportError as exc:
        raise ImportError(_INSTALL_HINTS[JAX]) from exc
    return jax_impl


def load_torch_impl():
    """Import and return lexicase.torch_impl, with a helpful error if torch is missing."""
    try:
        from . import torch_impl
    except ImportError as exc:
        raise ImportError(_INSTALL_HINTS[TORCH]) from exc
    return torch_impl
