"""
Public API for lexicase selection.

Every function validates its inputs, picks a backend, and calls the matching
kernel. Backend choice follows the input array: a NumPy array gives a NumPy
array back, a JAX array gives a JAX array back. Pass backend="numpy" or
backend="jax" to override.

Nothing here imports jax unless the JAX backend is actually used.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .backends import (
    JAX,
    TORCH,
    is_jax_array,
    is_torch_tensor,
    load_jax_impl,
    load_torch_impl,
    resolve_backend,
)
from .utils import validate_fitness_matrix, validate_selection_params

EPSILON_MODES = ("static", "semi-dynamic", "dynamic")


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


def _validate_shape(array) -> None:
    if array.ndim != 2:
        raise ValueError(
            f"Fitness matrix must be 2-dimensional, got {array.ndim}-dimensional"
        )
    if array.shape[0] == 0:
        raise ValueError("Fitness matrix must have at least one individual")
    if array.shape[1] == 0:
        raise ValueError("Fitness matrix must have at least one test case")


def _prepare(fitness_matrix, num_selected, seed, backend):
    """Validate inputs and resolve the backend.

    Returns:
        (array, backend_name) where array is a NumPy array for the NumPy
        backend and a JAX array for the JAX backend.
    """
    resolved = resolve_backend(backend, fitness_matrix)

    if resolved == JAX:
        array = load_jax_impl().jnp.asarray(fitness_matrix)
        _validate_shape(array)
    elif resolved == TORCH:
        array = load_torch_impl().torch.as_tensor(fitness_matrix)
        _validate_shape(array)
    else:
        array = validate_fitness_matrix(fitness_matrix)

    if num_selected < 0:
        raise ValueError("Number of selected individuals must be non-negative")
    if seed is not None and not is_jax_array(seed) and not is_torch_tensor(seed):
        validate_selection_params(num_selected, seed)

    return array, resolved


def _rng(seed):
    return np.random.default_rng(seed)


def _key(seed):
    """Turn a seed into a JAX PRNG key. Accepts an existing key unchanged."""
    jax_impl = load_jax_impl()
    if is_jax_array(seed):
        return seed
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    return jax_impl.jax.random.PRNGKey(int(seed))


def _on_device(value) -> bool:
    """True for a value already living on an accelerator, which must not be read."""
    return is_jax_array(value) or is_torch_tensor(value)


def _validate_case_weights(case_weights, n_cases):
    if case_weights is None:
        return None
    if _on_device(case_weights):
        if tuple(case_weights.shape) != (n_cases,):
            raise ValueError(
                f"case_weights must have length {n_cases}, "
                f"got shape {tuple(case_weights.shape)}"
            )
        return case_weights
    weights = np.asarray(case_weights, dtype=np.float64)
    if weights.shape != (n_cases,):
        raise ValueError(
            f"case_weights must have length {n_cases}, got shape {weights.shape}"
        )
    if np.any(weights <= 0):
        raise ValueError("All case weights must be positive")
    return weights


def _validate_epsilon(epsilon, n_cases):
    if _on_device(epsilon):
        if epsilon.ndim > 0 and epsilon.shape[0] != n_cases:
            raise ValueError(
                f"Epsilon array length ({epsilon.shape[0]}) must match "
                f"number of cases ({n_cases})"
            )
        return epsilon
    epsilon_array = np.asarray(epsilon)
    if epsilon_array.ndim > 0 and len(epsilon_array) != n_cases:
        raise ValueError(
            f"Epsilon array length ({epsilon_array.shape[0]}) must match "
            f"number of cases ({n_cases})"
        )
    if np.any(epsilon_array < 0):
        if epsilon_array.ndim == 0:
            raise ValueError("Epsilon must be non-negative")
        raise ValueError("All epsilon values must be non-negative")
    if epsilon_array.ndim == 0:
        return float(epsilon_array)
    return epsilon_array


def _as_jax(indices):
    """Move a NumPy result onto the JAX backend."""
    return load_jax_impl().jnp.asarray(indices)


def _as_torch(indices, like):
    """Move a NumPy result back onto the device of the input tensor."""
    torch = load_torch_impl().torch
    return torch.as_tensor(np.ascontiguousarray(indices), dtype=torch.long).to(like.device)


def lexicase_selection(
    fitness_matrix: ArrayLike,
    num_selected: int,
    seed: Optional[int] = None,
    elitism: int = 0,
    case_weights: Optional[ArrayLike] = None,
    backend: str = "auto",
) -> NDArray[np.intp]:
    """
    Lexicase selection.

    Each selection event shuffles the test cases and keeps only the candidates
    with the best fitness on each case in turn, until one candidate is left or
    the cases run out.

    Reference:
        Helmuth, T., Spector, L., and Matheson, J. (2015). Solving
        Uncompromising Problems with Lexicase Selection. IEEE Transactions on
        Evolutionary Computation 19(5), 630-643.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases). Higher is better.
        num_selected: Number of individuals to select
        seed: Random seed for reproducibility. On the JAX backend this may also
              be a PRNG key.
        elitism: Number of best individuals to always include (by total fitness)
        case_weights: Optional positive weight per case. Cases are then ordered
                      by weighted sampling without replacement instead of a
                      uniform shuffle, so heavier cases tend to come first.
        backend: "auto", "numpy", or "jax"

    Returns:
        Array of selected individual indices, matching the backend

    Raises:
        ValueError: If inputs are invalid
    """
    array, resolved = _prepare(fitness_matrix, num_selected, seed, backend)
    _validate_elitism(elitism, num_selected, array.shape[0])
    weights = _validate_case_weights(case_weights, array.shape[1])

    if resolved == JAX:
        return load_jax_impl().jax_lexicase_selection(
            array, num_selected, _key(seed), elitism, weights
        )

    if resolved == TORCH:
        return load_torch_impl().torch_lexicase_selection(
            array, num_selected, seed, elitism, weights
        )

    from .numpy_impl import numpy_lexicase_selection

    return numpy_lexicase_selection(array, num_selected, _rng(seed), elitism, weights)


def epsilon_lexicase_selection(
    fitness_matrix: ArrayLike,
    num_selected: int,
    epsilon: Optional[Union[float, ArrayLike]] = None,
    seed: Optional[int] = None,
    elitism: int = 0,
    mode: str = "semi-dynamic",
    case_weights: Optional[ArrayLike] = None,
    backend: str = "auto",
) -> NDArray[np.intp]:
    """
    Epsilon lexicase selection.

    Candidates within epsilon of the best performance on a case survive that
    case. The three modes differ in where the elite and the epsilon come from:

    - "static": both the elite and epsilon come from the whole population, and
      the fitness matrix is reduced to pass/fail once per call
    - "semi-dynamic": epsilon comes from the population, the elite comes from
      the current candidate pool. This is the paper's recommended default.
    - "dynamic": both the elite and epsilon come from the current pool, so an
      explicit epsilon is not used

    Reference:
        La Cava, W., Helmuth, T., Spector, L., and Moore, J. H. (2019). A
        probabilistic and multi-objective analysis of lexicase selection and
        epsilon-lexicase selection. Evolutionary Computation 27(3), 377-402.
        Algorithms 2, 3, and 4.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases). Higher is better.
        num_selected: Number of individuals to select
        epsilon: Scalar or per-case tolerance. If None, uses the median absolute
                 deviation of each case. Not allowed with mode="dynamic".
        seed: Random seed for reproducibility
        elitism: Number of best individuals to always include (by total fitness)
        mode: "static", "semi-dynamic", or "dynamic"
        case_weights: Optional positive weight per case for a non-uniform ordering
        backend: "auto", "numpy", or "jax"

    Returns:
        Array of selected individual indices, matching the backend

    Raises:
        ValueError: If inputs are invalid
    """
    if mode not in EPSILON_MODES:
        raise ValueError(f"Unknown epsilon mode {mode!r}, expected one of {EPSILON_MODES}")
    if mode == "dynamic" and epsilon is not None:
        raise ValueError(
            "Dynamic epsilon lexicase recomputes epsilon from the candidate pool, "
            "so an explicit epsilon cannot be used with mode='dynamic'"
        )

    array, resolved = _prepare(fitness_matrix, num_selected, seed, backend)
    _validate_elitism(elitism, num_selected, array.shape[0])
    weights = _validate_case_weights(case_weights, array.shape[1])

    if epsilon is not None:
        epsilon = _validate_epsilon(epsilon, array.shape[1])

    if resolved == JAX:
        jax_impl = load_jax_impl()
        tolerance = (
            jax_impl.jax_compute_mad_epsilon(array) if epsilon is None else epsilon
        )
        if mode == "dynamic":
            tolerance = 0.0
        return jax_impl.jax_epsilon_lexicase_selection(
            array, num_selected, tolerance, _key(seed), elitism, mode, weights
        )

    if resolved == TORCH:
        torch_impl = load_torch_impl()
        tolerance = (
            torch_impl.torch_compute_mad_epsilon(array) if epsilon is None else epsilon
        )
        if mode == "dynamic":
            tolerance = 0.0
        return torch_impl.torch_epsilon_lexicase_selection(
            array, num_selected, tolerance, seed, elitism, mode, weights
        )

    rng = _rng(seed)
    if epsilon is None:
        from .numpy_impl import numpy_epsilon_lexicase_selection_with_mad

        return numpy_epsilon_lexicase_selection_with_mad(
            array, num_selected, rng, elitism, mode, weights
        )

    from .numpy_impl import numpy_epsilon_lexicase_selection

    return numpy_epsilon_lexicase_selection(
        array, num_selected, epsilon, rng, elitism, mode, weights
    )


def downsample_lexicase_selection(
    fitness_matrix: ArrayLike,
    num_selected: int,
    downsample_size: int,
    seed: Optional[int] = None,
    elitism: int = 0,
    backend: str = "auto",
) -> NDArray[np.intp]:
    """
    Downsampled lexicase selection.

    Each selection event uses a fresh random subset of the cases, which cuts
    the work per event and increases the diversity of the selected parents.

    Reference:
        Hernandez, J. G., Lalejini, A., Dolson, E., and Ofria, C. (2019).
        Random subsampling improves performance in lexicase selection.
        GECCO '19 Companion, pp. 2028-2031.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases). Higher is better.
        num_selected: Number of individuals to select
        downsample_size: Number of cases to sample per selection event
        seed: Random seed for reproducibility
        elitism: Number of best individuals to always include (by total fitness)
        backend: "auto", "numpy", or "jax"

    Returns:
        Array of selected individual indices, matching the backend

    Raises:
        ValueError: If inputs are invalid
    """
    array, resolved = _prepare(fitness_matrix, num_selected, seed, backend)

    if downsample_size <= 0:
        raise ValueError("Downsample size must be positive")

    _validate_elitism(elitism, num_selected, array.shape[0])

    if resolved == JAX:
        return load_jax_impl().jax_downsample_lexicase_selection(
            array, num_selected, downsample_size, _key(seed), elitism
        )

    if resolved == TORCH:
        return load_torch_impl().torch_downsample_lexicase_selection(
            array, num_selected, downsample_size, seed, elitism
        )

    from .numpy_impl import numpy_downsample_lexicase_selection

    return numpy_downsample_lexicase_selection(
        array, num_selected, downsample_size, _rng(seed), elitism
    )


def informed_downsample_lexicase_selection(
    fitness_matrix: ArrayLike,
    num_selected: int,
    downsample_size: int,
    seed: Optional[int] = None,
    sample_rate: float = 0.01,
    threshold: Optional[Union[float, ArrayLike]] = None,
    elitism: int = 0,
    backend: str = "auto",
) -> NDArray[np.intp]:
    """
    Informed downsampled lexicase selection.

    Instead of sampling cases uniformly, this picks a case subset whose solve
    patterns are as different from each other as possible, using farthest first
    traversal over Hamming distances between cases. The subset is chosen once
    per call and reused for every selection event.

    Reference:
        Boldi, R., Briesch, M., Sobania, D., Lalejini, A., Helmuth, T.,
        Rothlauf, F., Ofria, C., and Spector, L. (2024). Informed Down-Sampled
        Lexicase Selection: Identifying Productive Training Cases for Efficient
        Problem Solving. Evolutionary Computation 32(4), 307-337.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases). Higher is better.
        num_selected: Number of individuals to select
        downsample_size: Number of cases to keep
        seed: Random seed for reproducibility
        sample_rate: Fraction of the population used to estimate solve patterns
        threshold: Pass/fail cutoff per case. If None, a matrix with at most two
                   distinct values is read as pass/fail directly, which is the
                   paper's own setting, and anything else falls back to a per-case
                   median split, which is a heuristic rather than the paper's rule.
                   Pass an explicit cutoff when you know your own pass mark. The
                   Torch backend never auto-detects, since that needs a host read.
        elitism: Number of best individuals to always include (by total fitness)
        backend: "auto", "numpy", or "jax"

    Returns:
        Array of selected individual indices, matching the backend

    Raises:
        ValueError: If inputs are invalid
    """
    array, resolved = _prepare(fitness_matrix, num_selected, seed, backend)

    if downsample_size <= 0:
        raise ValueError("Downsample size must be positive")
    if sample_rate <= 0 or sample_rate > 1:
        raise ValueError("Sample rate must be between 0 and 1")

    _validate_elitism(elitism, num_selected, array.shape[0])

    if resolved == JAX:
        from .numpy_impl import resolve_pass_threshold

        return load_jax_impl().jax_informed_downsample_lexicase_selection(
            array, num_selected, downsample_size, _key(seed), sample_rate,
            resolve_pass_threshold(array, threshold), elitism,
        )

    if resolved == TORCH:
        if threshold is None:
            raise ValueError(
                "informed downsampling needs a pass/fail cutoff, and the Torch "
                "backend will not infer one: reading the values to check would "
                "synchronize with the host, which is what this backend exists to "
                "avoid. Pass threshold=0.5 for 0/1 rewards, or your own cutoff. "
                "The NumPy and JAX backends infer it for you."
            )
        return load_torch_impl().torch_informed_downsample_lexicase_selection(
            array, num_selected, downsample_size, seed, sample_rate,
            threshold, elitism,
        )

    from .numpy_impl import numpy_informed_downsample_lexicase_selection

    return numpy_informed_downsample_lexicase_selection(
        array, num_selected, downsample_size, _rng(seed), sample_rate,
        threshold, elitism,
    )


def informed_downsample_cases(
    fitness_matrix: ArrayLike,
    downsample_size: int,
    seed: Optional[int] = None,
    sample_rate: float = 0.01,
    threshold: Optional[Union[float, ArrayLike]] = None,
    distances: Optional[ArrayLike] = None,
) -> tuple[NDArray[np.intp], NDArray[np.floating]]:
    """
    Pick an informed down-sample of cases, and return the distances behind it.

    This is the case-selection half of informed downsampling on its own, so you
    can implement the scheduled case distance computation of Boldi et al. (2024)
    Algorithm 2: recompute the distance matrix every k generations, but re-run
    the farthest first traversal every generation.

    Runs on the host on every backend, and always returns NumPy arrays.

    Reference:
        Boldi, R., Briesch, M., Sobania, D., Lalejini, A., Helmuth, T.,
        Rothlauf, F., Ofria, C., and Spector, L. (2024). Informed Down-Sampled
        Lexicase Selection: Identifying Productive Training Cases for Efficient
        Problem Solving. Evolutionary Computation 32(4), 307-337.
        Algorithms 1 and 2.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases). Higher is better.
        downsample_size: Number of cases to keep
        seed: Random seed for reproducibility
        sample_rate: The paper's rho. Fraction of the population evaluated on
                     every case to estimate the solve vectors.
        threshold: Pass/fail cutoff per case. If None, a two-valued matrix is read
                   as pass/fail directly and anything else falls back to a per-case
                   median split.
        distances: A distance matrix from an earlier call, to reuse instead of
                   recomputing. This is what makes the k schedule possible.

    Returns:
        (case_indices, distances), both NumPy arrays

    Example:
        >>> cases, distances = informed_downsample_cases(fitness, 10, seed=0)
        >>> # next generation, reuse the distances and redraw the sample
        >>> cases, _ = informed_downsample_cases(
        ...     fitness, 10, seed=1, distances=distances
        ... )
    """
    from .numpy_impl import (
        _compute_case_distances,
        _farthest_first_traversal,
        resolve_pass_threshold,
    )

    if is_torch_tensor(fitness_matrix):
        fitness_matrix = fitness_matrix.detach().cpu()
    array = validate_fitness_matrix(np.asarray(fitness_matrix))

    if downsample_size <= 0:
        raise ValueError("Downsample size must be positive")
    if sample_rate <= 0 or sample_rate > 1:
        raise ValueError("Sample rate must be between 0 and 1")

    rng = _rng(seed)
    n_individuals, n_cases = array.shape

    if distances is None:
        cutoff = resolve_pass_threshold(array, threshold)
        n_samples = max(1, int(n_individuals * sample_rate))
        sample_indices = rng.choice(n_individuals, size=n_samples, replace=False)
        distances = _compute_case_distances(array, sample_indices, cutoff)
    else:
        distances = np.asarray(distances)
        if distances.shape != (n_cases, n_cases):
            raise ValueError(
                f"distances must have shape ({n_cases}, {n_cases}), "
                f"got {distances.shape}"
            )

    cases = _farthest_first_traversal(
        distances, min(downsample_size, n_cases), rng
    )
    return cases, distances


def batch_lexicase_selection(
    fitness_matrix: ArrayLike,
    num_selected: int,
    batch_size: int,
    seed: Optional[int] = None,
    threshold: Optional[float] = None,
    elitism: int = 0,
    backend: str = "auto",
) -> NDArray[np.intp]:
    """
    Batch lexicase selection.

    Cases are shuffled and cut into consecutive batches. Each batch filters the
    pool on mean fitness over that batch, so the batch size tunes selection
    pressure: a batch size of 1 is standard lexicase, a batch size covering
    every case is elitist selection on mean fitness.

    Reference:
        Aenugu, S. and Spector, L. (2019). Lexicase Selection in Learning
        Classifier Systems. GECCO '19, pp. 356-364. Algorithm 2.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases). Higher is better.
        num_selected: Number of individuals to select
        batch_size: Number of cases per batch
        seed: Random seed for reproducibility
        threshold: If given, candidates survive a batch when their mean fitness
                   on it is strictly greater than this value, which is the
                   paper's pseudocode. If None, candidates survive when they are
                   elite on the batch, which is the paper's prose description and
                   does not assume fitness lies on any particular scale.
        elitism: Number of best individuals to always include (by total fitness)
        backend: "auto", "numpy", or "jax"

    Returns:
        Array of selected individual indices, matching the backend

    Raises:
        ValueError: If inputs are invalid
    """
    array, resolved = _prepare(fitness_matrix, num_selected, seed, backend)

    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    _validate_elitism(elitism, num_selected, array.shape[0])

    if resolved == JAX:
        return load_jax_impl().jax_batch_lexicase_selection(
            array, num_selected, batch_size, _key(seed), threshold, elitism
        )

    if resolved == TORCH:
        return load_torch_impl().torch_batch_lexicase_selection(
            array, num_selected, batch_size, seed, threshold, elitism
        )

    from .numpy_impl import numpy_batch_lexicase_selection

    return numpy_batch_lexicase_selection(
        array, num_selected, batch_size, _rng(seed), threshold, elitism
    )


def cohort_lexicase_selection(
    fitness_matrix: ArrayLike,
    num_selected: int,
    num_cohorts: int,
    seed: Optional[int] = None,
    elitism: int = 0,
    backend: str = "auto",
) -> NDArray[np.intp]:
    """
    Cohort lexicase selection.

    The population and the case set are each randomly split into num_cohorts
    cohorts. Cohort k of the population competes only within itself, judged
    only by cohort k of the cases. Every case is used somewhere each call, but
    each individual is only ever compared on 1/num_cohorts of them.

    Reference:
        Hernandez, J. G., Lalejini, A., Dolson, E., and Ofria, C. (2019).
        Random subsampling improves performance in lexicase selection.
        GECCO '19 Companion, pp. 2028-2031. Section 4.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases). Higher is better.
        num_selected: Number of individuals to select
        num_cohorts: Number of cohorts
        seed: Random seed for reproducibility
        elitism: Number of best individuals to always include (by total fitness)
        backend: "auto", "numpy", or "jax"

    Returns:
        Array of selected individual indices, matching the backend

    Raises:
        ValueError: If inputs are invalid
    """
    array, resolved = _prepare(fitness_matrix, num_selected, seed, backend)

    if num_cohorts <= 0:
        raise ValueError("Number of cohorts must be positive")
    if num_cohorts > array.shape[0]:
        raise ValueError("Number of cohorts cannot exceed number of individuals")
    if num_cohorts > array.shape[1]:
        raise ValueError("Number of cohorts cannot exceed number of cases")

    _validate_elitism(elitism, num_selected, array.shape[0])

    if resolved == JAX:
        return load_jax_impl().jax_cohort_lexicase_selection(
            array, num_selected, num_cohorts, _key(seed), elitism
        )

    if resolved == TORCH:
        return load_torch_impl().torch_cohort_lexicase_selection(
            array, num_selected, num_cohorts, seed, elitism
        )

    from .numpy_impl import numpy_cohort_lexicase_selection

    return numpy_cohort_lexicase_selection(
        array, num_selected, num_cohorts, _rng(seed), elitism
    )


def plexicase_selection(
    fitness_matrix: ArrayLike,
    num_selected: int,
    seed: Optional[int] = None,
    alpha: float = 1.0,
    epsilon: Optional[Union[float, ArrayLike]] = None,
    elitism: int = 0,
    backend: str = "auto",
) -> NDArray[np.intp]:
    """
    Probabilistic lexicase selection (plexicase).

    Builds an explicit approximation of the lexicase selection distribution
    once, then draws all parents from it. Individuals outside the Pareto set
    boundaries get probability zero.

    There is no native JAX or Torch kernel for this one. Finding the Pareto set
    boundaries needs a data-dependent number of candidates, which cannot be done
    without leaving the accelerator. With a JAX array or a Torch tensor in, the
    NumPy kernel runs on the host and the result is moved back to the input's
    backend and device. That means this function does synchronize, so it is the
    one selection method here that does not belong in a GPU inner loop.

    Reference:
        Ding, L., Pantridge, E., and Spector, L. (2023). Probabilistic Lexicase
        Selection. GECCO '23, pp. 1073-1081.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases). Higher is better.
        num_selected: Number of individuals to select
        seed: Random seed for reproducibility
        alpha: Temperature on the selection distribution. 1.0 is the paper's
               default, 0.0 makes it uniform over the Pareto set boundaries,
               larger values concentrate it on the most elite individuals.
        epsilon: Optional per-case tolerance for epsilon-plexicase
        elitism: Number of best individuals to always include (by total fitness)
        backend: "auto", "numpy", or "jax"

    Returns:
        Array of selected individual indices, matching the backend

    Raises:
        ValueError: If inputs are invalid
    """
    array, resolved = _prepare(fitness_matrix, num_selected, seed, backend)
    _validate_elitism(elitism, num_selected, array.shape[0])

    if epsilon is not None:
        epsilon = _validate_epsilon(epsilon, array.shape[1])

    from .numpy_impl import numpy_plexicase_selection

    host = np.asarray(array.cpu() if resolved == TORCH else array)
    selected = numpy_plexicase_selection(
        host, num_selected, _rng(seed), alpha, epsilon, elitism
    )
    if resolved == JAX:
        return _as_jax(selected)
    if resolved == TORCH:
        return _as_torch(selected, array)
    return selected


def plexicase_probabilities(
    fitness_matrix: ArrayLike,
    alpha: float = 1.0,
    epsilon: Optional[Union[float, ArrayLike]] = None,
) -> NDArray[np.floating]:
    """
    The plexicase selection distribution over the population.

    Useful on its own for measuring how concentrated lexicase selection is on a
    given population, without drawing any parents.

    Reference:
        Ding, L., Pantridge, E., and Spector, L. (2023). Probabilistic Lexicase
        Selection. GECCO '23, pp. 1073-1081. Equations 1 to 4.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases). Higher is better.
        alpha: Temperature on the distribution
        epsilon: Optional per-case tolerance

    Returns:
        NumPy array of length n_individuals summing to 1
    """
    if is_torch_tensor(fitness_matrix):
        fitness_matrix = fitness_matrix.detach().cpu()
    array = validate_fitness_matrix(np.asarray(fitness_matrix))
    if epsilon is not None:
        epsilon = _validate_epsilon(epsilon, array.shape[1])

    from .numpy_impl import numpy_plexicase_probabilities

    return numpy_plexicase_probabilities(array, alpha, epsilon)


def dalex_selection(
    fitness_matrix: ArrayLike,
    num_selected: int,
    seed: Optional[int] = None,
    particularity_pressure: float = 20.0,
    relaxed: bool = False,
    elitism: int = 0,
    backend: str = "auto",
) -> NDArray[np.intp]:
    """
    Diversely aggregated lexicase selection (DALex).

    Every selection event draws importance scores from a normal distribution,
    softmaxes them into case weights, and takes the individual with the best
    weighted mean fitness. The whole thing is one matrix multiply, which makes
    it the fastest variant here by a wide margin.

    Reference:
        Ni, A., Ding, L., and Spector, L. (2024). DALex: Lexicase-like Selection
        via Diverse Aggregation. EuroGP 2024, LNCS 14631, pp. 90-107. Algorithm 1.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases). Higher is better.
        num_selected: Number of individuals to select
        seed: Random seed for reproducibility
        particularity_pressure: Standard deviation of the importance scores.
                                Larger values behave more like standard lexicase.
                                The paper uses 20 for program synthesis and 3 for
                                symbolic regression, and suggests around 200 to
                                approximate standard lexicase.
        relaxed: Standardize each case before aggregating, which is how the
                 paper emulates epsilon lexicase
        elitism: Number of best individuals to always include (by total fitness)
        backend: "auto", "numpy", or "jax"

    Returns:
        Array of selected individual indices, matching the backend

    Raises:
        ValueError: If inputs are invalid
    """
    array, resolved = _prepare(fitness_matrix, num_selected, seed, backend)
    _validate_elitism(elitism, num_selected, array.shape[0])

    if particularity_pressure < 0:
        raise ValueError("Particularity pressure must be non-negative")

    if resolved == JAX:
        return load_jax_impl().jax_dalex_selection(
            array, num_selected, _key(seed), particularity_pressure, relaxed, elitism
        )

    if resolved == TORCH:
        return load_torch_impl().torch_dalex_selection(
            array, num_selected, seed, particularity_pressure, relaxed, elitism
        )

    from .numpy_impl import numpy_dalex_selection

    return numpy_dalex_selection(
        array, num_selected, _rng(seed), particularity_pressure, relaxed, elitism
    )
