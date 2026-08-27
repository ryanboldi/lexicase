"""
Lexicase selection for evolutionary computation.

Fast, vectorized implementations of lexicase selection and its variants, with
a NumPy backend and an optional JAX backend.

Importing this package never imports jax. Backend choice follows the input
array type, and can be forced with the backend= argument on every selection
function.

Usage:
    import numpy as np
    from lexicase import lexicase_selection

    fitness = np.random.rand(100, 20)  # 100 individuals, 20 test cases
    selected = lexicase_selection(fitness, num_selected=50, seed=42)
"""

from .backends import is_jax_array, jax_is_available
from .dispatch import (
    batch_lexicase_selection,
    cohort_lexicase_selection,
    dalex_selection,
    downsample_lexicase_selection,
    epsilon_lexicase_selection,
    informed_downsample_lexicase_selection,
    lexicase_selection,
    plexicase_probabilities,
    plexicase_selection,
)

__version__ = "0.4.0"
__all__ = [
    "lexicase_selection",
    "epsilon_lexicase_selection",
    "downsample_lexicase_selection",
    "informed_downsample_lexicase_selection",
    "batch_lexicase_selection",
    "cohort_lexicase_selection",
    "plexicase_selection",
    "plexicase_probabilities",
    "dalex_selection",
    "is_jax_array",
    "jax_is_available",
]
