"""
Lexicase selection library for evolutionary computation.

This library provides implementations of lexicase selection and its variants
with support for both NumPy and JAX backends.
"""

from .base import lexicase_selection
from .epsilon import epsilon_lexicase_selection
from .downsample import downsample_lexicase_selection
from .backend import set_backend, get_backend

__version__ = "0.1.0"
__all__ = [
    "lexicase_selection", 
    "epsilon_lexicase_selection", 
    "downsample_lexicase_selection",
    "set_backend", 
    "get_backend"
] 