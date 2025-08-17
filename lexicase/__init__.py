"""
Lexicase selection library for evolutionary computation.

This library provides implementations of lexicase selection and its variants
with automatic dispatch based on array type (NumPy or JAX).
"""

# Automatic dispatch API
from .dispatch import (
    lexicase_selection, 
    epsilon_lexicase_selection, 
    downsample_lexicase_selection
)

# Direct access to implementation modules (for advanced users)
from . import jax_impl, numpy_impl

# Legacy imports for backward compatibility
from .base import lexicase_selection as _legacy_lexicase_selection
from .epsilon import epsilon_lexicase_selection as _legacy_epsilon_lexicase_selection
from .downsample import downsample_lexicase_selection as _legacy_downsample_lexicase_selection

__version__ = "0.1.1"
__all__ = [
    "lexicase_selection", 
    "epsilon_lexicase_selection", 
    "downsample_lexicase_selection",
    "jax_impl",
    "numpy_impl"
] 