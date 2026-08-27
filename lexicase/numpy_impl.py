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

EPSILON_MODES = ("static", "semi-dynamic", "dynamic")


def sanitize(fitness_matrix: ArrayLike) -> NDArray[np.floating]:
    """Turn NaN into the worst possible value on its case.

    A NaN means the individual loses that case to anyone who scored a number
    there, and ties with anyone else who is NaN. Infinities are left alone.
    """
    array = np.asarray(fitness_matrix)
    if array.dtype.kind != "f" or not np.isnan(array).any():
        return array
    return np.where(np.isnan(array), -np.inf, array)


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


def _mad(values: NDArray[np.floating]) -> float:
    """Median absolute deviation of a 1-D array."""
    return float(np.median(np.abs(values - np.median(values))))


def _lexicase_select_one(
    fitness_matrix: NDArray[np.floating],
    case_order: NDArray[np.intp],
    rng: np.random.Generator,
    epsilon: Optional[NDArray[np.floating]] = None,
    dynamic_epsilon: bool = False,
) -> int:
    """Perform one lexicase selection event.

    Args:
        fitness_matrix: NumPy array of shape (n_individuals, n_cases)
        case_order: Shuffled order of test case indices
        rng: NumPy random number generator
        epsilon: Optional tolerance values per case for epsilon lexicase
        dynamic_epsilon: If True, recompute epsilon as the MAD of the current
                         candidate pool on each case (dynamic epsilon lexicase)

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

        if dynamic_epsilon:
            best_mask = case_fitness >= (max_fitness - _mad(case_fitness))
        elif epsilon is not None:
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


def _case_order(
    n_cases: int,
    rng: np.random.Generator,
    case_weights: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.intp]:
    """Draw a case ordering, uniform or weighted.

    Weighted orderings use the Efraimidis and Spirakis (2006) exponential
    key scheme, so a case with twice the weight is twice as likely to come
    first, and the same holds recursively for the rest of the order.
    """
    if case_weights is None:
        return rng.permutation(n_cases)
    keys = rng.random(n_cases) ** (1.0 / case_weights)
    return np.argsort(-keys).astype(np.intp)


def numpy_lexicase_selection(
    fitness_matrix: NDArray[np.floating],
    num_selected: int,
    rng: np.random.Generator,
    elitism: int = 0,
    case_weights: Optional[NDArray[np.floating]] = None,
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
    fitness_matrix = sanitize(fitness_matrix)
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
        case_order = _case_order(n_cases, rng, case_weights)
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
    mode: str = "semi-dynamic",
    case_weights: Optional[NDArray[np.floating]] = None,
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
    fitness_matrix = sanitize(fitness_matrix)
    if num_selected == 0:
        return np.array([], dtype=np.intp)

    n_individuals, n_cases = fitness_matrix.shape

    if mode not in EPSILON_MODES:
        raise ValueError(f"Unknown epsilon mode {mode!r}, expected one of {EPSILON_MODES}")

    # Handle epsilon - ensure it's the right shape
    epsilon_values = np.broadcast_to(epsilon, (n_cases,)).astype(np.float64)

    filter_matrix = fitness_matrix
    if mode == "static":
        case_best = np.max(fitness_matrix, axis=0)
        passes = fitness_matrix >= (case_best - epsilon_values)[None, :]
        filter_matrix = passes.astype(np.float64)
        epsilon_values = None

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
        case_order = _case_order(n_cases, rng, case_weights)
        selected[selection_idx] = _lexicase_select_one(
            filter_matrix,
            case_order,
            rng,
            epsilon=epsilon_values,
            dynamic_epsilon=(mode == "dynamic"),
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
    mode: str = "semi-dynamic",
    case_weights: Optional[NDArray[np.floating]] = None,
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
        fitness_matrix, num_selected, epsilon_values, rng, elitism, mode, case_weights
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
    fitness_matrix = sanitize(fitness_matrix)
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
        sampled_cases = rng.choice(n_cases, size=actual_downsample_size, replace=False)
        case_order = rng.permutation(actual_downsample_size)

        selected[selection_idx] = _lexicase_select_one(
            fitness_matrix, sampled_cases[case_order], rng, epsilon=None
        )
        selection_idx += 1

    return selected


def resolve_pass_threshold(
    fitness_matrix: ArrayLike,
    threshold: Optional[Union[float, ArrayLike]] = None,
) -> Optional[Union[float, ArrayLike]]:
    """Work out the pass/fail cutoff for informed downsampling.

    Boldi et al. (2024) define case distances over binary solve vectors, so the
    fitness matrix has to be reduced to "solved" and "not solved" first. When the
    matrix has at most two distinct values it already is pass/fail, and the cutoff
    is the midpoint between them. That covers 0/1 fitness and the negated 0/-1
    errors this package tells you to pass in.

    Anything with more than two distinct values is not pass/fail data, and there
    is no scale-free way to guess where "solved" begins, so this returns None and
    the caller falls back to a per-case median split. Pass an explicit threshold
    when you know your own pass mark.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases)
        threshold: A caller-supplied cutoff, returned unchanged if it is not None

    Returns:
        The cutoff to compare against with >, or None to use the median heuristic
    """
    if threshold is not None:
        return threshold

    low = float(fitness_matrix.min())
    high = float(fitness_matrix.max())
    if low == high:
        return low
    if not bool(((fitness_matrix == low) | (fitness_matrix == high)).all()):
        return None
    return (low + high) / 2.0


def _compute_case_distances(
    fitness_matrix: NDArray[np.floating],
    sample_indices: NDArray[np.intp],
    threshold: Optional[Union[float, NDArray[np.floating]]] = None,
) -> NDArray[np.floating]:
    """
    Compute pairwise distances between test cases based on solve patterns.

    The distance between two cases is the Hamming distance between their binary
    solve vectors, following Boldi et al. (2024) Section 3.

    Args:
        fitness_matrix: Full fitness matrix (n_individuals, n_cases)
        sample_indices: Indices of individuals to use for distance calculation
        threshold: Threshold for pass/fail. If None, uses median per case, which
                   is a heuristic for continuous fitness and not the paper's rule.
                   Callers should resolve it with resolve_pass_threshold first.

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

    solved = solve_matrix.astype(np.float64)
    mismatch = solved.T @ (1.0 - solved)
    return mismatch + mismatch.T


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
    fitness_matrix = sanitize(fitness_matrix)
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

    threshold = resolve_pass_threshold(fitness_matrix, threshold)

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


def numpy_batch_lexicase_selection(
    fitness_matrix: NDArray[np.floating],
    num_selected: int,
    batch_size: int,
    rng: np.random.Generator,
    threshold: Optional[float] = None,
    elitism: int = 0,
) -> NDArray[np.intp]:
    """
    NumPy-based batch lexicase selection.

    Cases are shuffled and cut into consecutive batches of `batch_size`. Each
    batch filters the candidate pool by mean fitness over the batch, so larger
    batches mean weaker filtering per step and more survivors.

    Reference:
        Aenugu, S. and Spector, L. (2019). Lexicase Selection in Learning
        Classifier Systems. GECCO '19, pp. 356-364. Algorithm 2.

    Args:
        fitness_matrix: NumPy array of shape (n_individuals, n_cases)
        num_selected: Number of individuals to select
        batch_size: Number of cases per batch
        rng: NumPy random number generator
        threshold: If given, candidates survive a batch when their mean fitness
                   on it is strictly greater than this value, which is the
                   pseudocode in the paper. If None, candidates survive when
                   they are elite on the batch, which is the prose description
                   and the reading that does not assume a fitness scale.
        elitism: Number of best individuals to always include (by total fitness)

    Returns:
        NumPy array of selected individual indices
    """
    fitness_matrix = sanitize(fitness_matrix)
    if num_selected == 0:
        return np.array([], dtype=np.intp)

    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    n_individuals, n_cases = fitness_matrix.shape

    selected = np.empty(num_selected, dtype=np.intp)
    selection_idx = 0

    if elitism > 0:
        selected[:elitism] = _select_elites(fitness_matrix, elitism)
        selection_idx = elitism

    while selection_idx < num_selected:
        case_order = rng.permutation(n_cases)
        candidates = np.arange(n_individuals)

        for start in range(0, n_cases, batch_size):
            if len(candidates) <= 1:
                break
            batch = case_order[start : start + batch_size]
            scores = np.mean(fitness_matrix[np.ix_(candidates, batch)], axis=1)
            if threshold is None:
                keep = scores == np.max(scores)
            else:
                keep = scores > threshold
                if not np.any(keep):
                    continue
            candidates = candidates[keep]

        if len(candidates) == 1:
            selected[selection_idx] = candidates[0]
        else:
            selected[selection_idx] = candidates[rng.choice(len(candidates))]
        selection_idx += 1

    return selected


def numpy_cohort_lexicase_selection(
    fitness_matrix: NDArray[np.floating],
    num_selected: int,
    num_cohorts: int,
    rng: np.random.Generator,
    elitism: int = 0,
) -> NDArray[np.intp]:
    """
    NumPy-based cohort lexicase selection.

    Both the population and the case set are randomly partitioned into
    `num_cohorts` equally sized cohorts. Population cohort k competes only
    against itself, arbitrated only by case cohort k. Every case is used
    somewhere, but each individual only ever sees 1/num_cohorts of them.

    Reference:
        Hernandez, J. G., Lalejini, A., Dolson, E., and Ofria, C. (2019).
        Random subsampling improves performance in lexicase selection.
        GECCO '19 Companion, pp. 2028-2031. Section 4.

    Args:
        fitness_matrix: NumPy array of shape (n_individuals, n_cases)
        num_selected: Number of individuals to select
        num_cohorts: Number of cohorts to split the population and cases into
        rng: NumPy random number generator
        elitism: Number of best individuals to always include (by total fitness)

    Returns:
        NumPy array of selected individual indices
    """
    fitness_matrix = sanitize(fitness_matrix)
    if num_selected == 0:
        return np.array([], dtype=np.intp)

    n_individuals, n_cases = fitness_matrix.shape

    if num_cohorts <= 0:
        raise ValueError("Number of cohorts must be positive")
    if num_cohorts > n_individuals:
        raise ValueError("Number of cohorts cannot exceed number of individuals")
    if num_cohorts > n_cases:
        raise ValueError("Number of cohorts cannot exceed number of cases")

    selected = np.empty(num_selected, dtype=np.intp)
    selection_idx = 0

    if elitism > 0:
        selected[:elitism] = _select_elites(fitness_matrix, elitism)
        selection_idx = elitism

    individual_cohorts = np.array_split(rng.permutation(n_individuals), num_cohorts)
    case_cohorts = np.array_split(rng.permutation(n_cases), num_cohorts)

    remaining = num_selected - selection_idx
    per_cohort = np.full(num_cohorts, remaining // num_cohorts, dtype=int)
    per_cohort[: remaining % num_cohorts] += 1

    for cohort_idx in range(num_cohorts):
        members = individual_cohorts[cohort_idx]
        cases = case_cohorts[cohort_idx]
        submatrix = fitness_matrix[np.ix_(members, cases)]
        for _ in range(per_cohort[cohort_idx]):
            case_order = rng.permutation(len(cases))
            local = _lexicase_select_one(submatrix, case_order, rng, epsilon=None)
            selected[selection_idx] = members[local]
            selection_idx += 1

    return selected


def numpy_plexicase_probabilities(
    fitness_matrix: NDArray[np.floating],
    alpha: float = 1.0,
    epsilon: Optional[Union[float, NDArray[np.floating]]] = None,
) -> NDArray[np.floating]:
    """
    Approximate lexicase selection probabilities for every individual.

    Individuals outside the Pareto set boundaries get probability zero. The
    rest get a probability proportional to how often they are elite, averaged
    over cases, then sharpened or flattened by `alpha`.

    Reference:
        Ding, L., Pantridge, E., and Spector, L. (2023). Probabilistic Lexicase
        Selection. GECCO '23, pp. 1073-1081. Equations 1 to 4.

    Args:
        fitness_matrix: NumPy array of shape (n_individuals, n_cases)
        alpha: Temperature on the final distribution. 1.0 leaves it unchanged,
               0.0 makes it uniform over the Pareto set boundaries, larger
               values concentrate it on the most elite individuals.
        epsilon: Optional per-case tolerance for epsilon-relaxed elitism and
                 epsilon-domination.

    Returns:
        NumPy array of length n_individuals summing to 1
    """
    n_individuals, n_cases = fitness_matrix.shape

    if alpha < 0:
        raise ValueError("Alpha must be non-negative")

    if epsilon is None:
        tolerance = np.zeros(n_cases, dtype=np.float64)
    else:
        tolerance = np.broadcast_to(epsilon, (n_cases,)).astype(np.float64)

    case_best = np.max(fitness_matrix, axis=0)
    is_elite = fitness_matrix >= (case_best - tolerance)[None, :]
    elitism_count = np.sum(is_elite, axis=1)

    candidates = np.flatnonzero(elitism_count > 0)
    kept = candidates[~_dominated_mask(fitness_matrix[candidates], tolerance)]

    probabilities = np.zeros(n_individuals, dtype=np.float64)
    if len(kept) == 0:
        probabilities[:] = 1.0 / n_individuals
        return probabilities

    density = np.where(is_elite[kept], elitism_count[kept][:, None], 0.0)
    column_totals = np.sum(density, axis=0)
    per_case = np.divide(
        density,
        column_totals[None, :],
        out=np.zeros_like(density),
        where=column_totals[None, :] > 0,
    )
    scores = np.sum(per_case, axis=1) / n_cases

    if alpha != 1.0:
        scores = scores**alpha

    total = np.sum(scores)
    if total <= 0:
        scores = np.ones(len(kept), dtype=np.float64)
        total = float(len(kept))

    probabilities[kept] = scores / total
    return probabilities


def _dominated_mask(
    fitness: NDArray[np.floating],
    tolerance: NDArray[np.floating],
    chunk: int = 128,
) -> NDArray[np.bool_]:
    """Mark rows of `fitness` that some other row dominates.

    With zero tolerance this is strict Pareto dominance, so individuals with
    identical fitness vectors both survive. With a tolerance it is the
    epsilon-domination of Ding et al. (2023) Definition 3.11, made
    antisymmetric so that mutually epsilon-dominating rows both survive.
    """
    n = fitness.shape[0]
    dominated = np.zeros(n, dtype=bool)
    if n < 2:
        return dominated

    strict = not np.any(tolerance)
    for start in range(0, n, chunk):
        block = fitness[start : start + chunk]
        if strict:
            ge = np.all(fitness[:, None, :] >= block[None, :, :], axis=2)
            gt = np.any(fitness[:, None, :] > block[None, :, :], axis=2)
            beats = ge & gt
        else:
            shifted = fitness - tolerance[None, :]
            beats = np.all(shifted[:, None, :] >= block[None, :, :], axis=2)
            reverse = np.all(
                (block - tolerance[None, :])[None, :, :] >= fitness[:, None, :], axis=2
            )
            beats = beats & ~reverse
        np.fill_diagonal(beats[start : start + block.shape[0]], False)
        dominated[start : start + block.shape[0]] = np.any(beats, axis=0)

    return dominated


def numpy_plexicase_selection(
    fitness_matrix: NDArray[np.floating],
    num_selected: int,
    rng: np.random.Generator,
    alpha: float = 1.0,
    epsilon: Optional[Union[float, NDArray[np.floating]]] = None,
    elitism: int = 0,
) -> NDArray[np.intp]:
    """
    NumPy-based probabilistic lexicase selection (plexicase).

    Reference:
        Ding, L., Pantridge, E., and Spector, L. (2023). Probabilistic Lexicase
        Selection. GECCO '23, pp. 1073-1081.

    Args:
        fitness_matrix: NumPy array of shape (n_individuals, n_cases)
        num_selected: Number of individuals to select
        rng: NumPy random number generator
        alpha: Temperature on the selection distribution
        epsilon: Optional per-case tolerance for epsilon-plexicase
        elitism: Number of best individuals to always include (by total fitness)

    Returns:
        NumPy array of selected individual indices
    """
    fitness_matrix = sanitize(fitness_matrix)
    if num_selected == 0:
        return np.array([], dtype=np.intp)

    selected = np.empty(num_selected, dtype=np.intp)
    selection_idx = 0

    if elitism > 0:
        selected[:elitism] = _select_elites(fitness_matrix, elitism)
        selection_idx = elitism

    probabilities = numpy_plexicase_probabilities(fitness_matrix, alpha, epsilon)
    remaining = num_selected - selection_idx
    if remaining > 0:
        draws = rng.choice(len(probabilities), size=remaining, p=probabilities)
        selected[selection_idx:] = draws

    return selected


def numpy_dalex_selection(
    fitness_matrix: NDArray[np.floating],
    num_selected: int,
    rng: np.random.Generator,
    particularity_pressure: float = 20.0,
    relaxed: bool = False,
    elitism: int = 0,
) -> NDArray[np.intp]:
    """
    NumPy-based diversely aggregated lexicase selection (DALex).

    Each selection event draws importance scores from N(0, particularity
    pressure), softmaxes them into case weights, and picks the individual with
    the best weighted mean fitness. Large particularity pressure concentrates
    the weights on one case and approaches standard lexicase; small values
    approach a plain fitness average.

    Reference:
        Ni, A., Ding, L., and Spector, L. (2024). DALex: Lexicase-like
        Selection via Diverse Aggregation. EuroGP 2024, LNCS 14631,
        pp. 90-107. Algorithm 1.

    Args:
        fitness_matrix: NumPy array of shape (n_individuals, n_cases)
        num_selected: Number of individuals to select
        rng: NumPy random number generator
        particularity_pressure: Standard deviation of the importance scores.
                                The paper uses 20 for program synthesis and 3
                                for symbolic regression, and suggests values
                                around 200 to approximate standard lexicase.
        relaxed: Standardize each case before aggregating, which is how the
                 paper emulates epsilon lexicase
        elitism: Number of best individuals to always include (by total fitness)

    Returns:
        NumPy array of selected individual indices
    """
    fitness_matrix = sanitize(fitness_matrix)
    if num_selected == 0:
        return np.array([], dtype=np.intp)

    if particularity_pressure < 0:
        raise ValueError("Particularity pressure must be non-negative")

    n_individuals, n_cases = fitness_matrix.shape

    scores_matrix = np.asarray(fitness_matrix, dtype=np.float64)
    if relaxed:
        spread = np.std(scores_matrix, axis=0)
        spread = np.where(spread > 0, spread, 1.0)
        scores_matrix = (scores_matrix - np.mean(scores_matrix, axis=0)) / spread

    selected = np.empty(num_selected, dtype=np.intp)
    selection_idx = 0

    if elitism > 0:
        selected[:elitism] = _select_elites(fitness_matrix, elitism)
        selection_idx = elitism

    remaining = num_selected - selection_idx
    if remaining > 0:
        importance = rng.normal(0.0, particularity_pressure, size=(remaining, n_cases))
        importance -= np.max(importance, axis=1, keepdims=True)
        weights = np.exp(importance)
        weights /= np.sum(weights, axis=1, keepdims=True)
        aggregated = scores_matrix @ weights.T
        selected[selection_idx:] = np.argmax(aggregated, axis=0)

    return selected
