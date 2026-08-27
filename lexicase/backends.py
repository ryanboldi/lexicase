"""
Backend selection for lexicase selection.

This module never imports jax at module load time. A JAX array cannot exist
unless the caller has already imported jax, so `sys.modules` is enough to
detect one. Importing `lexicase` therefore never pulls in jax, even when jax
is installed.
"""

from __future__ import annotations

import sys
from typing import Any

NUMPY = "numpy"
JAX = "jax"
AUTO = "auto"

_JAX_INSTALL_HINT = (
    "The JAX backend requires jax. Install it with: pip install 'lexicase[jax]'"
)


def is_jax_array(x: Any) -> bool:
    """Return True if x is a JAX array, without importing jax."""
    jax = sys.modules.get("jax")
    array_type = getattr(jax, "Array", None)
    if array_type is None:
        return False
    return isinstance(x, array_type)


def jax_is_available() -> bool:
    """Return True if jax can be imported. Imports jax as a side effect."""
    try:
        import jax  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(backend: str, fitness_matrix: Any) -> str:
    """Resolve the requested backend to "numpy" or "jax".

    Args:
        backend: "auto", "numpy", or "jax"
        fitness_matrix: The input array, used when backend is "auto"

    Returns:
        Either "numpy" or "jax"
    """
    if backend == AUTO:
        return JAX if is_jax_array(fitness_matrix) else NUMPY
    if backend in (NUMPY, JAX):
        return backend
    raise ValueError(f"Unknown backend {backend!r}, expected 'auto', 'numpy', or 'jax'")


def load_jax_impl():
    """Import and return lexicase.jax_impl, with a helpful error if jax is missing."""
    try:
        from . import jax_impl
    except ImportError as exc:
        raise ImportError(_JAX_INSTALL_HINT) from exc
    return jax_impl
