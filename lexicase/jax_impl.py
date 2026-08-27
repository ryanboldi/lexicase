"""
Pure JAX implementations of lexicase selection algorithms.

Every function here takes a JAX PRNG key instead of a seed and returns a JAX
array. The kernels use static shapes throughout, so they are jittable with the
size arguments marked static and vmappable over a batch of fitness matrices.
See the backends section of the README for the exact list.

This module imports jax. Nothing in the package imports it at load time, so
`import lexicase` works with numpy alone.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

MIN_EPSILON = 1e-10


def sanitize(fitness_matrix):
    """Turn NaN into the worst possible value on its case.

    A NaN means the individual loses that case to anyone who scored a number
    there, and ties with anyone else who is NaN. Infinities are left alone.
    This is one elementwise pass and stays inside jit.
    """
    array = jnp.asarray(fitness_matrix)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        return array.astype(jnp.float32)
    return jnp.where(jnp.isnan(array), -jnp.inf, array)


def _random_from_mask(key, mask):
    """Pick one index uniformly at random from the True entries of mask."""
    noise = jax.random.uniform(key, mask.shape)
    return jnp.argmax(jnp.where(mask, noise, -1.0))


def _uniform_case_order(key, n_cases):
    return jax.random.permutation(key, n_cases)


def _weighted_case_order(key, case_weights):
    keys = jax.random.uniform(key, case_weights.shape) ** (1.0 / case_weights)
    return jnp.argsort(-keys)


def _filter_by_case(fitness, mask, case_idx, epsilon, dynamic_epsilon):
    values = jnp.where(mask, fitness[:, case_idx], -jnp.inf)
    best = jnp.max(values)
    if dynamic_epsilon:
        pool = jnp.where(mask, fitness[:, case_idx], jnp.nan)
        tolerance = jnp.nanmedian(jnp.abs(pool - jnp.nanmedian(pool)))
    elif epsilon is None:
        tolerance = 0.0
    else:
        tolerance = epsilon[case_idx]
    survivors = mask & (values >= best - tolerance)
    return jnp.where(jnp.sum(mask) <= 1, mask, survivors)


def _select_one(fitness, case_order, key, epsilon=None, dynamic_epsilon=False):
    """One lexicase selection event over the given case order."""
    n_individuals = fitness.shape[0]

    def step(mask, case_idx):
        return _filter_by_case(fitness, mask, case_idx, epsilon, dynamic_epsilon), None

    mask, _ = lax.scan(step, jnp.ones(n_individuals, dtype=bool), case_order)
    return _random_from_mask(key, mask)


def _select_elites(fitness, elitism):
    return jnp.argsort(jnp.sum(fitness, axis=1))[-elitism:]


def _with_elites(fitness, elitism, body_key, num_selected, body):
    """Run `body` for the non-elite slots and prepend the elites."""
    if elitism <= 0:
        return body(body_key, num_selected)
    elites = _select_elites(fitness, elitism).astype(jnp.int32)
    if num_selected == elitism:
        return elites
    return jnp.concatenate([elites, body(body_key, num_selected - elitism)])


def _run_events(fitness, num_events, key, order_fn, epsilon, dynamic_epsilon):
    order_keys = jax.random.split(key, num_events)
    tie_keys = jax.random.split(jax.random.fold_in(key, 1), num_events)

    def one(order_key, tie_key):
        order = order_fn(order_key)
        return _select_one(fitness, order, tie_key, epsilon, dynamic_epsilon)

    return jax.vmap(one)(order_keys, tie_keys).astype(jnp.int32)


def jax_compute_mad_epsilon(fitness_matrix):
    """Median absolute deviation of each case, floored at MIN_EPSILON."""
    medians = jnp.median(fitness_matrix, axis=0)
    deviations = jnp.abs(fitness_matrix - medians[None, :])
    return jnp.maximum(jnp.median(deviations, axis=0), MIN_EPSILON)


def jax_lexicase_selection(
    fitness_matrix,
    num_selected: int,
    key,
    elitism: int = 0,
    case_weights=None,
):
    """
    fitness_matrix = sanitize(fitness_matrix)
    JAX lexicase selection.

    Jittable with num_selected and elitism static.

    Args:
        fitness_matrix: JAX array of shape (n_individuals, n_cases). Higher is better.
        num_selected: Number of individuals to select
        key: JAX PRNG key
        elitism: Number of best individuals to always include (by total fitness)
        case_weights: Optional per-case weights for a non-uniform case ordering

    Returns:
        JAX int32 array of selected individual indices
    """
    fitness_matrix = sanitize(fitness_matrix)
    if num_selected == 0:
        return jnp.array([], dtype=jnp.int32)

    n_cases = fitness_matrix.shape[1]
    if case_weights is None:
        order_fn = lambda k: _uniform_case_order(k, n_cases)  # noqa: E731
    else:
        weights = jnp.asarray(case_weights)
        order_fn = lambda k: _weighted_case_order(k, weights)  # noqa: E731

    return _with_elites(
        fitness_matrix,
        elitism,
        key,
        num_selected,
        lambda k, n: _run_events(fitness_matrix, n, k, order_fn, None, False),
    )


def jax_epsilon_lexicase_selection(
    fitness_matrix,
    num_selected: int,
    epsilon,
    key,
    elitism: int = 0,
    mode: str = "semi-dynamic",
    case_weights=None,
):
    """
    JAX epsilon lexicase selection.

    Reference:
        La Cava, W., Helmuth, T., Spector, L., and Moore, J. H. (2019). A
        probabilistic and multi-objective analysis of lexicase selection and
        epsilon-lexicase selection. Evolutionary Computation 27(3), 377-402.

    Jittable with num_selected, elitism, and mode static.

    Args:
        fitness_matrix: JAX array of shape (n_individuals, n_cases)
        num_selected: Number of individuals to select
        epsilon: Scalar or per-case tolerance. Ignored when mode is "dynamic".
        key: JAX PRNG key
        elitism: Number of best individuals to always include
        mode: "static", "semi-dynamic", or "dynamic"
        case_weights: Optional per-case weights for a non-uniform case ordering

    Returns:
        JAX int32 array of selected individual indices
    """
    fitness_matrix = sanitize(fitness_matrix)
    if num_selected == 0:
        return jnp.array([], dtype=jnp.int32)

    n_cases = fitness_matrix.shape[1]
    epsilon_values = jnp.broadcast_to(jnp.asarray(epsilon, dtype=jnp.float32), (n_cases,))

    filter_matrix = fitness_matrix
    if mode == "static":
        case_best = jnp.max(fitness_matrix, axis=0)
        filter_matrix = (fitness_matrix >= (case_best - epsilon_values)[None, :]).astype(
            fitness_matrix.dtype
        )
        epsilon_values = None
    elif mode == "dynamic":
        epsilon_values = None
    elif mode != "semi-dynamic":
        raise ValueError(f"Unknown epsilon mode {mode!r}")

    if case_weights is None:
        order_fn = lambda k: _uniform_case_order(k, n_cases)  # noqa: E731
    else:
        weights = jnp.asarray(case_weights)
        order_fn = lambda k: _weighted_case_order(k, weights)  # noqa: E731

    return _with_elites(
        fitness_matrix,
        elitism,
        key,
        num_selected,
        lambda k, n: _run_events(
            filter_matrix, n, k, order_fn, epsilon_values, mode == "dynamic"
        ),
    )


def jax_downsample_lexicase_selection(
    fitness_matrix,
    num_selected: int,
    downsample_size: int,
    key,
    elitism: int = 0,
):
    """
    JAX downsampled lexicase selection.

    Each selection event draws its own random subset of cases without
    replacement, which is both the subset and its order in one draw.

    Reference:
        Hernandez, J. G., Lalejini, A., Dolson, E., and Ofria, C. (2019).
        Random subsampling improves performance in lexicase selection.
        GECCO '19 Companion, pp. 2028-2031.

    Jittable with num_selected, downsample_size, and elitism static.
    """
    fitness_matrix = sanitize(fitness_matrix)
    if num_selected == 0:
        return jnp.array([], dtype=jnp.int32)

    n_cases = fitness_matrix.shape[1]
    size = min(downsample_size, n_cases)
    order_fn = lambda k: jax.random.choice(  # noqa: E731
        k, n_cases, shape=(size,), replace=False
    )

    return _with_elites(
        fitness_matrix,
        elitism,
        key,
        num_selected,
        lambda k, n: _run_events(fitness_matrix, n, k, order_fn, None, False),
    )


def _farthest_first_traversal(distances, downsample_size: int, key):
    """Farthest first traversal over a case distance matrix."""
    n_cases = distances.shape[0]
    if downsample_size >= n_cases:
        return jnp.arange(n_cases, dtype=jnp.int32)

    first_key, loop_key = jax.random.split(key)
    first = jax.random.randint(first_key, (), 0, n_cases)
    picks = jnp.zeros(downsample_size, dtype=jnp.int32).at[0].set(first)
    chosen = jnp.zeros(n_cases, dtype=bool).at[first].set(True)

    def body(i, carry):
        picks, chosen, min_distance, key = carry
        key, tie_key = jax.random.split(key)
        scores = jnp.where(chosen, -jnp.inf, min_distance)
        pick = _random_from_mask(tie_key, scores == jnp.max(scores))
        return (
            picks.at[i].set(pick.astype(jnp.int32)),
            chosen.at[pick].set(True),
            jnp.minimum(min_distance, distances[pick]),
            key,
        )

    picks, _, _, _ = lax.fori_loop(
        1, downsample_size, body, (picks, chosen, distances[first], loop_key)
    )
    return picks


def _informative_cases(fitness_matrix, downsample_size, key, sample_rate, threshold):
    n_individuals = fitness_matrix.shape[0]
    n_samples = max(1, int(n_individuals * sample_rate))

    sample_key, traversal_key = jax.random.split(key)
    sample_indices = jax.random.choice(
        sample_key, n_individuals, shape=(n_samples,), replace=False
    )
    sampled = fitness_matrix[sample_indices, :]

    if threshold is None:
        cutoff = jnp.median(sampled, axis=0)
    else:
        cutoff = jnp.asarray(threshold)
    solved = (sampled > cutoff).astype(jnp.float32)

    mismatch = solved.T @ (1.0 - solved)
    return _farthest_first_traversal(mismatch + mismatch.T, downsample_size, traversal_key)


def jax_informed_downsample_lexicase_selection(
    fitness_matrix,
    num_selected: int,
    downsample_size: int,
    key,
    sample_rate: float = 0.01,
    threshold=None,
    elitism: int = 0,
):
    """
    JAX informed downsampled lexicase selection.

    The case subset is chosen once per call by farthest first traversal over
    Hamming distances between case solve patterns, then reused for every
    selection event, matching the NumPy backend.

    Reference:
        Boldi, R., Briesch, M., Sobania, D., Lalejini, A., Helmuth, T.,
        Rothlauf, F., Ofria, C., and Spector, L. (2024). Informed Down-Sampled
        Lexicase Selection: Identifying Productive Training Cases for Efficient
        Problem Solving. Evolutionary Computation 32(4), 307-337.

    Jittable with num_selected, downsample_size, sample_rate, and elitism static.
    """
    fitness_matrix = sanitize(fitness_matrix)
    if num_selected == 0:
        return jnp.array([], dtype=jnp.int32)

    n_cases = fitness_matrix.shape[1]
    size = min(downsample_size, n_cases)

    case_key, selection_key = jax.random.split(key)
    cases = _informative_cases(
        fitness_matrix, size, case_key, sample_rate, threshold
    )
    submatrix = fitness_matrix[:, cases]
    order_fn = lambda k: _uniform_case_order(k, size)  # noqa: E731

    return _with_elites(
        fitness_matrix,
        elitism,
        selection_key,
        num_selected,
        lambda k, n: _run_events(submatrix, n, k, order_fn, None, False),
    )


def jax_batch_lexicase_selection(
    fitness_matrix,
    num_selected: int,
    batch_size: int,
    key,
    threshold: Optional[float] = None,
    elitism: int = 0,
):
    """
    JAX batch lexicase selection.

    Reference:
        Aenugu, S. and Spector, L. (2019). Lexicase Selection in Learning
        Classifier Systems. GECCO '19, pp. 356-364. Algorithm 2.

    Jittable with num_selected, batch_size, threshold, and elitism static.
    The batch loop is unrolled at trace time, so very large case counts with a
    small batch size make compilation slow.
    """
    fitness_matrix = sanitize(fitness_matrix)
    if num_selected == 0:
        return jnp.array([], dtype=jnp.int32)

    n_individuals, n_cases = fitness_matrix.shape
    starts = list(range(0, n_cases, batch_size))

    def select_one(order_key, tie_key):
        order = jax.random.permutation(order_key, n_cases)
        mask = jnp.ones(n_individuals, dtype=bool)
        for start in starts:
            batch = lax.dynamic_slice(
                order, (start,), (min(batch_size, n_cases - start),)
            )
            scores = jnp.where(
                mask, jnp.mean(fitness_matrix[:, batch], axis=1), -jnp.inf
            )
            if threshold is None:
                survivors = mask & (scores >= jnp.max(scores))
            else:
                survivors = mask & (scores > threshold)
                survivors = jnp.where(jnp.any(survivors), survivors, mask)
            mask = jnp.where(jnp.sum(mask) <= 1, mask, survivors)
        return _random_from_mask(tie_key, mask)

    def body(k, n):
        order_keys = jax.random.split(k, n)
        tie_keys = jax.random.split(jax.random.fold_in(k, 1), n)
        return jax.vmap(select_one)(order_keys, tie_keys).astype(jnp.int32)

    return _with_elites(fitness_matrix, elitism, key, num_selected, body)


def jax_cohort_lexicase_selection(
    fitness_matrix,
    num_selected: int,
    num_cohorts: int,
    key,
    elitism: int = 0,
):
    """
    JAX cohort lexicase selection.

    Reference:
        Hernandez, J. G., Lalejini, A., Dolson, E., and Ofria, C. (2019).
        Random subsampling improves performance in lexicase selection.
        GECCO '19 Companion, pp. 2028-2031. Section 4.

    Jittable with num_selected, num_cohorts, and elitism static.
    """
    fitness_matrix = sanitize(fitness_matrix)
    if num_selected == 0:
        return jnp.array([], dtype=jnp.int32)

    n_individuals, n_cases = fitness_matrix.shape
    if num_cohorts > n_individuals or num_cohorts > n_cases:
        raise ValueError("Number of cohorts cannot exceed individuals or cases")

    def body(body_key, remaining):
        pop_key, case_key, select_key = jax.random.split(body_key, 3)
        pop_order = jax.random.permutation(pop_key, n_individuals)
        case_order = jax.random.permutation(case_key, n_cases)
        cohort_keys = jax.random.split(select_key, num_cohorts)

        pop_bounds = _split_bounds(n_individuals, num_cohorts)
        case_bounds = _split_bounds(n_cases, num_cohorts)
        counts = _split_counts(remaining, num_cohorts)

        parts = []
        for cohort in range(num_cohorts):
            if counts[cohort] == 0:
                continue
            members = lax.dynamic_slice(pop_order, (pop_bounds[cohort][0],),
                                        (pop_bounds[cohort][1],))
            cases = lax.dynamic_slice(case_order, (case_bounds[cohort][0],),
                                      (case_bounds[cohort][1],))
            submatrix = fitness_matrix[jnp.ix_(members, cases)]
            order_fn = lambda k, size=case_bounds[cohort][1]: _uniform_case_order(k, size)
            local = _run_events(
                submatrix, counts[cohort], cohort_keys[cohort], order_fn, None, False
            )
            parts.append(members[local].astype(jnp.int32))
        return jnp.concatenate(parts)

    return _with_elites(fitness_matrix, elitism, key, num_selected, body)


def _split_bounds(total, parts):
    """Offsets and sizes matching numpy.array_split."""
    base, extra = divmod(total, parts)
    bounds = []
    offset = 0
    for i in range(parts):
        size = base + (1 if i < extra else 0)
        bounds.append((offset, size))
        offset += size
    return bounds


def _split_counts(total, parts):
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def jax_dalex_selection(
    fitness_matrix,
    num_selected: int,
    key,
    particularity_pressure: float = 20.0,
    relaxed: bool = False,
    elitism: int = 0,
):
    """
    JAX diversely aggregated lexicase selection (DALex).

    Reference:
        Ni, A., Ding, L., and Spector, L. (2024). DALex: Lexicase-like
        Selection via Diverse Aggregation. EuroGP 2024, LNCS 14631,
        pp. 90-107. Algorithm 1.

    Jittable with num_selected, relaxed, and elitism static. This is the one
    variant that is a single matrix multiply, so it is by far the fastest on
    accelerators.
    """
    fitness_matrix = sanitize(fitness_matrix)
    if num_selected == 0:
        return jnp.array([], dtype=jnp.int32)

    n_cases = fitness_matrix.shape[1]
    scores_matrix = fitness_matrix
    if relaxed:
        spread = jnp.std(scores_matrix, axis=0)
        spread = jnp.where(spread > 0, spread, 1.0)
        scores_matrix = (scores_matrix - jnp.mean(scores_matrix, axis=0)) / spread

    def body(body_key, remaining):
        importance = (
            jax.random.normal(body_key, (remaining, n_cases)) * particularity_pressure
        )
        weights = jax.nn.softmax(importance, axis=1)
        return jnp.argmax(scores_matrix @ weights.T, axis=0).astype(jnp.int32)

    return _with_elites(fitness_matrix, elitism, key, num_selected, body)
