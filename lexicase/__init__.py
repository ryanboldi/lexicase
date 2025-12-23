"""
Lexicase selection library for evolutionary computation.

This library provides fast, vectorized implementations of lexicase selection
and its variants using NumPy.

Usage:
    import numpy as np
    from lexicase import lexicase_selection

    fitness = np.random.rand(100, 20)  # 100 individuals, 20 test cases
    selected = lexicase_selection(fitness, num_selected=50, seed=42)
"""

from .dispatch import (
    lexicase_selection,
    epsilon_lexicase_selection,
    downsample_lexicase_selection,
    informed_downsample_lexicase_selection
)

__version__ = "0.3.0"
__all__ = [
    "lexicase_selection",
    "epsilon_lexicase_selection",
    "downsample_lexicase_selection",
    "informed_downsample_lexicase_selection",
]
