"""
Pure NumPy implementations of lexicase selection algorithms.

These functions are optimized for NumPy arrays and provide efficient
CPU-based lexicase selection without external dependencies.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .utils import MIN_EPSILON


def _select_elites(
    fitness_matrix: NDArray[np.floating],
    elitism: int,
) -> NDArray[np.intp]:
    """Select elite individuals by total fitness.

    Args:
        fitness_matrix: NumPy array of shape (n_individuals, n_cases)
        elitism: Number of elite individuals to select

    Returns:
        Array of elite individual indices
    """
    if elitism <= 0:
        return np.array([], dtype=np.intp)

    total_fitness = np.sum(fitness_matrix, axis=1)
    elite_indices = np.argsort(total_fitness)[-elitism:]
    return elite_indices.astype(np.intp)


def _lexicase_select_one(
    fitness_matrix: NDArray[np.floating],
    case_order: NDArray[np.intp],
    rng: np.random.Generator,
    epsilon: Optional[NDArray[np.floating]] = None,
) -> int:
    """Perform one lexicase selection event.

    Args:
        fitness_matrix: NumPy array of shape (n_individuals, n_cases)
        case_order: Shuffled order of test case indices
        rng: NumPy random number generator
        epsilon: Optional tolerance values per case for epsilon lexicase

    Returns:
        Index of selected individual
    """
    n_individuals = fitness_matrix.shape[0]
    candidates = np.arange(n_individuals)

    for case_idx in case_order:
        if len(candidates) <= 1:
            break

        case_fitness = fitness_matrix[candidates, case_idx]
        max_fitness = np.max(case_fitness)

        if epsilon is not None:
            case_epsilon = epsilon[case_idx]
            best_mask = case_fitness >= (max_fitness - case_epsilon)
        else:
            best_mask = case_fitness == max_fitness

        candidates = candidates[best_mask]

    if len(candidates) == 1:
        return int(candidates[0])
    else:
        chosen_idx = rng.choice(len(candidates))
        return int(candidates[chosen_idx])


def numpy_lexicase_selection(
    fitness_matrix: NDArray[np.floating],
    num_selected: int,
    rng: np.random.Generator,
    elitism: int = 0,
) -> NDArray[np.intp]:
    """
    NumPy-based lexicase selection implementation.

    Args:
        fitness_matrix: NumPy array of shape (n_individuals, n_cases)
                       Higher values indicate better performance.
        num_selected: Number of individuals to select (int)
        rng: NumPy random number generator (from np.random.default_rng())
        elitism: Number of best individuals to always include (by total fitness)

    Returns:
        NumPy array of selected individual indices
    """
    if num_selected == 0:
        return np.array([], dtype=np.intp)

    n_individuals, n_cases = fitness_matrix.shape

    # Pre-allocate result array
    selected = np.empty(num_selected, dtype=np.intp)
    selection_idx = 0

    # Handle elitism
    if elitism > 0:
        elite_indices = _select_elites(fitness_matrix, elitism)
        selected[:elitism] = elite_indices
        selection_idx = elitism

    # Perform regular lexicase selection for remaining slots
    while selection_idx < num_selected:
        case_order = rng.permutation(n_cases)
        selected[selection_idx] = _lexicase_select_one(
            fitness_matrix, case_order, rng, epsilon=None
        )
        selection_idx += 1

    return selected


def numpy_epsilon_lexicase_selection(
    fitness_matrix: NDArray[np.floating],
    num_selected: int,
    epsilon: Union[float, NDArray[np.floating]],
    rng: np.random.Generator,
    elitism: int = 0,
) -> NDArray[np.intp]:
    """
    NumPy-based epsilon lexicase selection implementation.

    Args:
        fitness_matrix: NumPy array of shape (n_individuals, n_cases)
        num_selected: Number of individuals to select
        epsilon: Tolerance value(s). Can be scalar or array of length n_cases
        rng: NumPy random number generator
        elitism: Number of best individuals to always include (by total fitness)

    Returns:
        NumPy array of selected individual indices
    """
    if num_selected == 0:
        return np.array([], dtype=np.intp)

    n_individuals, n_cases = fitness_matrix.shape

    # Handle epsilon - ensure it's the right shape
    epsilon_values = np.broadcast_to(epsilon, (n_cases,)).astype(np.float64)

    # Pre-allocate result array
    selected = np.empty(num_selected, dtype=np.intp)
    selection_idx = 0

    # Handle elitism
    if elitism > 0:
        elite_indices = _select_elites(fitness_matrix, elitism)
        selected[:elitism] = elite_indices
        selection_idx = elitism

    # Perform selection for remaining slots
    while selection_idx < num_selected:
        case_order = rng.permutation(n_cases)
        selected[selection_idx] = _lexicase_select_one(
            fitness_matrix, case_order, rng, epsilon=epsilon_values
        )
        selection_idx += 1

    return selected


def numpy_compute_mad_epsilon(
    fitness_matrix: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Compute Median Absolute Deviation (MAD) for each test case using NumPy.

    Args:
        fitness_matrix: NumPy array of shape (n_individuals, n_cases)

    Returns:
        NumPy array of MAD values for each test case
    """
    # Calculate median for each case (column)
    case_medians = np.median(fitness_matrix, axis=0)

    # Calculate absolute deviations from median for each case
    abs_deviations = np.abs(fitness_matrix - case_medians[None, :])

    # Calculate median of absolute deviations for each case
    mad_values = np.median(abs_deviations, axis=0)

    # Handle case where MAD is 0 (all values identical) by using a small default
    mad_values = np.maximum(mad_values, MIN_EPSILON)

    return mad_values


def numpy_epsilon_lexicase_selection_with_mad(
    fitness_matrix: NDArray[np.floating],
    num_selected: int,
    rng: np.random.Generator,
    elitism: int = 0,
) -> NDArray[np.intp]:
    """
    NumPy epsilon lexicase selection using MAD-based adaptive epsilon.

    Args:
        fitness_matrix: NumPy array of shape (n_individuals, n_cases)
        num_selected: Number of individuals to select
        rng: NumPy random number generator
        elitism: Number of best individuals to always include (by total fitness)

    Returns:
        NumPy array of selected individual indices
    """
    # Compute MAD-based epsilon values
    epsilon_values = numpy_compute_mad_epsilon(fitness_matrix)

    # Use epsilon lexicase with computed epsilon
    return numpy_epsilon_lexicase_selection(
        fitness_matrix, num_selected, epsilon_values, rng, elitism
    )


def numpy_downsample_lexicase_selection(
    fitness_matrix: NDArray[np.floating],
    num_selected: int,
    downsample_size: int,
    rng: np.random.Generator,
    elitism: int = 0,
) -> NDArray[np.intp]:
    """
    NumPy-based downsampled lexicase selection implementation.

    Args:
        fitness_matrix: NumPy array of shape (n_individuals, n_cases)
        num_selected: Number of individuals to select
        downsample_size: Number of test cases to randomly sample for each selection
        rng: NumPy random number generator
        elitism: Number of best individuals to always include (by total fitness)

    Returns:
        NumPy array of selected individual indices
    """
    if num_selected == 0:
        return np.array([], dtype=np.intp)

    if downsample_size <= 0:
        raise ValueError("Downsample size must be positive")

    n_individuals, n_cases = fitness_matrix.shape
    actual_downsample_size = min(downsample_size, n_cases)

    # Pre-allocate result array
    selected = np.empty(num_selected, dtype=np.intp)
    selection_idx = 0

    # Handle elitism
    if elitism > 0:
        elite_indices = _select_elites(fitness_matrix, elitism)
        selected[:elitism] = elite_indices
        selection_idx = elitism

    # Perform selection for remaining slots
    while selection_idx < num_selected:
        # Randomly sample test cases for this selection
        sampled_cases = rng.choice(n_cases, size=actual_downsample_size, replace=False)

        # Create submatrix with only sampled cases
        submatrix = fitness_matrix[:, sampled_cases]

        # Shuffle case order for the submatrix
        case_order = rng.permutation(actual_downsample_size)

        # Perform lexicase selection on the submatrix
        selected[selection_idx] = _lexicase_select_one(
            submatrix, case_order, rng, epsilon=None
        )
        selection_idx += 1

    return selected


def _compute_case_distances(
    fitness_matrix: NDArray[np.floating],
    sample_indices: NDArray[np.intp],
    threshold: Optional[Union[float, NDArray[np.floating]]] = None,
) -> NDArray[np.floating]:
    """
    Compute pairwise distances between test cases based on solve patterns.

    Args:
        fitness_matrix: Full fitness matrix (n_individuals, n_cases)
        sample_indices: Indices of individuals to use for distance calculation
        threshold: Optional threshold for pass/fail. If None, uses median per case.

    Returns:
        Distance matrix of shape (n_cases, n_cases)
    """
    # Get sampled fitness values
    sampled_fitness = fitness_matrix[sample_indices, :]
    n_samples, n_cases = sampled_fitness.shape

    # Create binary solve matrix
    if threshold is None:
        # Use median as threshold for each case
        thresholds = np.median(sampled_fitness, axis=0)
        solve_matrix = sampled_fitness > thresholds[None, :]
    elif np.isscalar(threshold):
        # Use single threshold for all
        solve_matrix = sampled_fitness > threshold
    else:
        # Use per-case thresholds
        solve_matrix = sampled_fitness > np.asarray(threshold)[None, :]

    # Compute Hamming distances between cases using vectorized operations
    # Each column is a case's solve pattern across sampled individuals
    distances = np.zeros((n_cases, n_cases), dtype=np.float64)
    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            # Hamming distance: count differences in solve patterns
            distance = np.sum(solve_matrix[:, i] != solve_matrix[:, j])
            distances[i, j] = distance
            distances[j, i] = distance

    return distances


def _farthest_first_traversal(
    distances: NDArray[np.floating],
    downsample_size: int,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    """
    Select cases using Farthest First Traversal algorithm.

    Args:
        distances: Pairwise distance matrix between cases (n_cases, n_cases)
        downsample_size: Number of cases to select
        rng: NumPy random number generator

    Returns:
        Array of selected case indices
    """
    n_cases = distances.shape[0]

    # Handle edge cases
    if downsample_size >= n_cases:
        return np.arange(n_cases, dtype=np.intp)

    selected: list[int] = []
    remaining = list(range(n_cases))

    # Randomly select first case
    first_idx = int(rng.choice(remaining))
    selected.append(first_idx)
    remaining.remove(first_idx)

    # Iteratively add cases that maximize minimum distance to selected cases
    while len(selected) < downsample_size and remaining:
        min_distances = []

        for case_idx in remaining:
            # Find minimum distance to any selected case
            min_dist = min(distances[case_idx, s] for s in selected)
            min_distances.append(min_dist)

        # Find cases with maximum minimum distance
        min_distances_arr = np.array(min_distances)
        max_min_dist = np.max(min_distances_arr)

        # Handle ties randomly
        candidates = [
            remaining[i]
            for i in range(len(remaining))
            if min_distances_arr[i] == max_min_dist
        ]

        if candidates:
            chosen = int(rng.choice(candidates))
            selected.append(chosen)
            remaining.remove(chosen)
        else:
            # If all distances are 0, randomly select from remaining
            chosen = int(rng.choice(remaining))
            selected.append(chosen)
            remaining.remove(chosen)

    return np.array(selected, dtype=np.intp)


def numpy_informed_downsample_lexicase_selection(
    fitness_matrix: NDArray[np.floating],
    num_selected: int,
    downsample_size: int,
    rng: np.random.Generator,
    sample_rate: float = 0.01,
    threshold: Optional[Union[float, NDArray[np.floating]]] = None,
    elitism: int = 0,
) -> NDArray[np.intp]:
    """
    NumPy-based informed downsampled lexicase selection implementation.

    Uses population statistics to select informative test cases rather than
    random sampling.

    Args:
        fitness_matrix: NumPy array of shape (n_individuals, n_cases)
        num_selected: Number of individuals to select
        downsample_size: Number of test cases to select for each selection
        rng: NumPy random number generator
        sample_rate: Fraction of population to sample for distance calculation
        threshold: Optional threshold for pass/fail. If None, uses median.
        elitism: Number of best individuals to always include (by total fitness)

    Returns:
        NumPy array of selected individual indices
    """
    if num_selected == 0:
        return np.array([], dtype=np.intp)

    if downsample_size <= 0:
        raise ValueError("Downsample size must be positive")

    n_individuals, n_cases = fitness_matrix.shape
    actual_downsample_size = min(downsample_size, n_cases)

    # Pre-allocate result array
    selected = np.empty(num_selected, dtype=np.intp)
    selection_idx = 0

    # Handle elitism
    if elitism > 0:
        elite_indices = _select_elites(fitness_matrix, elitism)
        selected[:elitism] = elite_indices
        selection_idx = elitism

    # Sample individuals for distance calculation
    n_samples = max(1, int(n_individuals * sample_rate))
    sample_indices = rng.choice(n_individuals, size=n_samples, replace=False)

    # Compute case distances based on sampled individuals
    distances = _compute_case_distances(fitness_matrix, sample_indices, threshold)

    # Select informative cases using Farthest First Traversal
    informative_cases = _farthest_first_traversal(distances, actual_downsample_size, rng)

    # Create submatrix with only informative cases
    submatrix = fitness_matrix[:, informative_cases]

    # Perform selection for remaining slots
    while selection_idx < num_selected:
        # Shuffle case order for the submatrix
        case_order = rng.permutation(actual_downsample_size)

        # Perform lexicase selection on the submatrix
        selected[selection_idx] = _lexicase_select_one(
            submatrix, case_order, rng, epsilon=None
        )
        selection_idx += 1

    return selected
