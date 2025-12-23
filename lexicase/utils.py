"""
Utility functions for lexicase selection.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Named constants for magic numbers
MIN_VARIANCE_THRESHOLD: float = 1e-10
MIN_EPSILON: float = 1e-10


def validate_fitness_matrix(fitness_matrix: ArrayLike) -> NDArray[np.floating]:
    """Validate fitness matrix format and content.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases)

    Returns:
        Validated NumPy array of shape (n_individuals, n_cases)

    Raises:
        ValueError: If matrix format is invalid
    """
    fitness_array = np.asarray(fitness_matrix)

    if fitness_array.ndim != 2:
        raise ValueError(
            f"Fitness matrix must be 2-dimensional, got {fitness_array.ndim}-dimensional"
        )

    if fitness_array.shape[0] == 0:
        raise ValueError("Fitness matrix must have at least one individual")

    if fitness_array.shape[1] == 0:
        raise ValueError("Fitness matrix must have at least one test case")

    return fitness_array


def validate_selection_params(num_selected: int, seed: Optional[int] = None) -> None:
    """Validate selection parameters.

    Args:
        num_selected: Number of individuals to select
        seed: Random seed

    Raises:
        ValueError: If parameters are invalid
    """
    if num_selected < 0:
        raise ValueError("Number of selected individuals must be non-negative")

    if seed is not None and not isinstance(seed, (int, np.integer)):
        raise ValueError("Seed must be an integer")


def shuffle_cases(num_cases: int, rng: np.random.Generator) -> NDArray[np.intp]:
    """Shuffle test case indices.

    Args:
        num_cases: Number of test cases
        rng: NumPy random number generator

    Returns:
        Array of shuffled case indices
    """
    return rng.permutation(num_cases)


def compute_case_variance(fitness_matrix: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute variance for each test case across individuals.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases)

    Returns:
        Array of variances for each test case
    """
    return np.var(fitness_matrix, axis=0)


def select_informative_cases(
    fitness_matrix: NDArray[np.floating],
    downsample_size: int,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    """Select informative test cases based on variance.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases)
        downsample_size: Number of cases to select
        rng: NumPy random number generator

    Returns:
        Array of selected case indices
    """
    variances = compute_case_variance(fitness_matrix)
    num_cases = fitness_matrix.shape[1]

    # If all variances are zero or very small, fall back to uniform sampling
    if np.all(variances < MIN_VARIANCE_THRESHOLD):
        case_indices = np.arange(num_cases)
        return rng.choice(case_indices, size=min(downsample_size, num_cases), replace=False)

    # Select cases with probability proportional to variance
    probabilities = variances / np.sum(variances)

    return rng.choice(
        num_cases,
        size=min(downsample_size, num_cases),
        replace=False,
        p=probabilities,
    )


def compute_mad_epsilon(fitness_matrix: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute Median Absolute Deviation (MAD) for each test case.

    MAD is calculated as the median of absolute deviations from the median
    for each test case across all individuals.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases)

    Returns:
        Array of MAD values for each test case
    """
    # Calculate median for each case (column) - more robust than mean
    case_medians = np.median(fitness_matrix, axis=0)

    # Calculate absolute deviations from median for each case
    abs_deviations = np.abs(fitness_matrix - case_medians[None, :])

    # Calculate median of absolute deviations for each case
    mad_values = np.median(abs_deviations, axis=0)

    # Handle case where MAD is 0 (all values identical) by using a small default
    mad_values = np.maximum(mad_values, MIN_EPSILON)

    return mad_values
