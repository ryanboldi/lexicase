"""
Dispatch layer for lexicase selection algorithms.

This module provides the public API for lexicase selection functions
with input validation.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .utils import validate_fitness_matrix, validate_selection_params


def _validate_elitism(elitism: int, num_selected: int, n_individuals: int) -> None:
    """Validate elitism parameter.

    Args:
        elitism: Number of elite individuals
        num_selected: Total number to select
        n_individuals: Number of individuals in population

    Raises:
        ValueError: If elitism is invalid
    """
    if elitism < 0:
        raise ValueError("Elitism must be non-negative")
    if elitism > num_selected:
        raise ValueError("Elitism cannot exceed num_selected")
    if elitism > n_individuals:
        raise ValueError("Elitism cannot exceed number of individuals")


def lexicase_selection(
    fitness_matrix: ArrayLike,
    num_selected: int,
    seed: Optional[int] = None,
    elitism: int = 0,
) -> NDArray[np.intp]:
    """
    Lexicase selection algorithm.

    Selects individuals based on their performance across multiple test cases.
    Each selection event shuffles the test cases and filters candidates
    by keeping only those with maximum fitness on each case in sequence.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases) containing
                       fitness values. Higher values indicate better performance.
        num_selected: Number of individuals to select
        seed: Random seed for reproducibility
        elitism: Number of best individuals to always include (by total fitness).
                 The remaining (num_selected - elitism) slots are filled via
                 standard lexicase selection. Default is 0 (no elitism).

    Returns:
        NumPy array of selected individual indices

    Raises:
        ValueError: If inputs are invalid
    """
    fitness_array = validate_fitness_matrix(fitness_matrix)
    validate_selection_params(num_selected, seed)
    _validate_elitism(elitism, num_selected, fitness_array.shape[0])

    from .numpy_impl import numpy_lexicase_selection

    rng = np.random.default_rng(seed)
    return numpy_lexicase_selection(fitness_array, num_selected, rng, elitism)


def epsilon_lexicase_selection(
    fitness_matrix: ArrayLike,
    num_selected: int,
    epsilon: Optional[Union[float, ArrayLike]] = None,
    seed: Optional[int] = None,
    elitism: int = 0,
) -> NDArray[np.intp]:
    """
    Epsilon lexicase selection algorithm.

    A relaxed version of lexicase selection where individuals within epsilon
    of the best performance on a case are kept as candidates.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases) containing
                       fitness values. Higher values indicate better performance.
        num_selected: Number of individuals to select
        epsilon: Tolerance value for "equal" performance. If None (default),
                uses Median Absolute Deviation (MAD) for each case.
                Can be a scalar (same epsilon for all cases) or array-like
                (different epsilon per case).
        seed: Random seed for reproducibility
        elitism: Number of best individuals to always include (by total fitness).
                 Default is 0 (no elitism).

    Returns:
        NumPy array of selected individual indices

    Raises:
        ValueError: If inputs are invalid
    """
    fitness_array = validate_fitness_matrix(fitness_matrix)
    validate_selection_params(num_selected, seed)
    _validate_elitism(elitism, num_selected, fitness_array.shape[0])

    rng = np.random.default_rng(seed)

    if epsilon is None:
        from .numpy_impl import numpy_epsilon_lexicase_selection_with_mad

        return numpy_epsilon_lexicase_selection_with_mad(
            fitness_array, num_selected, rng, elitism
        )
    else:
        from .numpy_impl import numpy_epsilon_lexicase_selection

        epsilon_np = np.asarray(epsilon)

        if epsilon_np.ndim > 0 and len(epsilon_np) != fitness_array.shape[1]:
            raise ValueError(
                f"Epsilon array length ({len(epsilon_np)}) must match "
                f"number of cases ({fitness_array.shape[1]})"
            )
        if np.any(epsilon_np < 0):
            if epsilon_np.ndim == 0:
                raise ValueError("Epsilon must be non-negative")
            else:
                raise ValueError("All epsilon values must be non-negative")

        return numpy_epsilon_lexicase_selection(
            fitness_array, num_selected, epsilon_np, rng, elitism
        )


def downsample_lexicase_selection(
    fitness_matrix: ArrayLike,
    num_selected: int,
    downsample_size: int,
    seed: Optional[int] = None,
    elitism: int = 0,
) -> NDArray[np.intp]:
    """
    Downsampled lexicase selection algorithm.

    A faster variant that randomly samples a subset of test cases for each
    selection event, reducing computational cost while maintaining diversity.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases) containing
                       fitness values. Higher values indicate better performance.
        num_selected: Number of individuals to select
        downsample_size: Number of test cases to randomly sample for each selection
        seed: Random seed for reproducibility
        elitism: Number of best individuals to always include (by total fitness).
                 Default is 0 (no elitism).

    Returns:
        NumPy array of selected individual indices

    Raises:
        ValueError: If inputs are invalid
    """
    fitness_array = validate_fitness_matrix(fitness_matrix)
    validate_selection_params(num_selected, seed)

    if downsample_size <= 0:
        raise ValueError("Downsample size must be positive")

    _validate_elitism(elitism, num_selected, fitness_array.shape[0])

    from .numpy_impl import numpy_downsample_lexicase_selection

    rng = np.random.default_rng(seed)
    return numpy_downsample_lexicase_selection(
        fitness_array, num_selected, downsample_size, rng, elitism
    )


def informed_downsample_lexicase_selection(
    fitness_matrix: ArrayLike,
    num_selected: int,
    downsample_size: int,
    seed: Optional[int] = None,
    sample_rate: float = 0.01,
    threshold: Optional[Union[float, ArrayLike]] = None,
    elitism: int = 0,
) -> NDArray[np.intp]:
    """
    Informed downsampled lexicase selection algorithm.

    Uses population statistics to select informative test cases that are maximally
    different from each other, rather than random sampling. This can improve
    problem-solving success by ensuring diverse test coverage.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases) containing
                       fitness values. Higher values indicate better performance.
        num_selected: Number of individuals to select
        downsample_size: Number of test cases to select for each selection
        seed: Random seed for reproducibility
        sample_rate: Fraction of population to sample for distance calculation (default 0.01)
        threshold: Optional threshold for pass/fail determination. If None (default),
                  uses median performance per case. Can be scalar or array per case.
        elitism: Number of best individuals to always include (by total fitness).
                 Default is 0 (no elitism).

    Returns:
        NumPy array of selected individual indices

    Raises:
        ValueError: If inputs are invalid
    """
    fitness_array = validate_fitness_matrix(fitness_matrix)
    validate_selection_params(num_selected, seed)

    if downsample_size <= 0:
        raise ValueError("Downsample size must be positive")

    if sample_rate <= 0 or sample_rate > 1:
        raise ValueError("Sample rate must be between 0 and 1")

    _validate_elitism(elitism, num_selected, fitness_array.shape[0])

    from .numpy_impl import numpy_informed_downsample_lexicase_selection

    rng = np.random.default_rng(seed)
    return numpy_informed_downsample_lexicase_selection(
        fitness_array,
        num_selected,
        downsample_size,
        rng,
        sample_rate,
        threshold,
        elitism,
    )
