"""
Pure PyTorch implementations of lexicase selection algorithms.

Written for the case where the fitness matrix is a reward tensor already on an
accelerator, for instance inside an RL trainer. Every kernel here preserves the
input tensor's device, and none of them synchronize with the host: no `.item()`,
no `.cpu()`, and no Python-level branching on tensor values. Selection events
are batched, so the per-case loop runs once for all of them at a fixed trip
count taken from the tensor's shape rather than from its contents.

Two consequences of never synchronizing:

- There is no early exit when every event has narrowed to one candidate. The
  loop always runs over every case. That trades a little wasted work for a
  guarantee of no stalls, which is the right trade on a GPU.
- `torch_plexicase_selection` is not implemented here. Finding the Pareto set
  boundaries needs a data-dependent number of candidates, which cannot be done
  without a host sync. The dispatch layer falls back to the NumPy kernel and
  moves the result back to the original device, and that call does sync.

NaN policy: a NaN fitness value is treated as the worst possible performance on
its case. See `sanitize`.

This module imports torch. Nothing else in the package does.
"""

from __future__ import annotations

from typing import Optional

import torch

MIN_EPSILON = 1e-10


def sanitize(fitness_matrix: torch.Tensor) -> torch.Tensor:
    """Promote to a floating dtype and turn NaN into the worst possible value.

    A NaN on a case means the individual loses that case to anyone who scored a
    number there, and ties with anyone else who is NaN. This is one elementwise
    pass and does not synchronize.
    """
    if not fitness_matrix.is_floating_point():
        return fitness_matrix.to(torch.float32)
    return torch.where(torch.isnan(fitness_matrix), float("-inf"), fitness_matrix)


def _generator(fitness_matrix: torch.Tensor, seed: Optional[int]) -> torch.Generator:
    generator = torch.Generator(device=fitness_matrix.device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(int(seed))
    return generator


def to_device(value, like):
    """Keep Python scalars as scalars, move anything array-shaped to like's device.

    Creating a tensor from host data is a host-to-device copy, which is the one
    thing here that stalls. Scalars stay scalars so they never cause one, and
    array-shaped arguments are moved once per call, outside the per-case loop.
    Pass them already on the device to avoid even that.
    """
    if value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(value, dtype=like.dtype).to(like.device)


def _random(shape, fitness_matrix, generator):
    return torch.rand(
        shape,
        generator=generator,
        device=fitness_matrix.device,
        dtype=fitness_matrix.dtype,
    )


def _case_orders(n_events, n_cases, fitness_matrix, generator, case_weights=None):
    """One independent case ordering per selection event, shape (n_events, n_cases)."""
    keys = _random((n_events, n_cases), fitness_matrix, generator)
    if case_weights is not None:
        keys = keys ** (1.0 / case_weights[None, :])
    return torch.argsort(keys, dim=1, descending=case_weights is not None)


def _random_from_mask(mask, fitness_matrix, generator):
    """Pick one True entry per row, uniformly at random."""
    noise = _random(mask.shape, fitness_matrix, generator)
    return torch.where(mask, noise, -1.0).argmax(dim=1)


def _pool_mad(values, mask):
    """Median absolute deviation of each row, over the masked entries only."""
    pool = torch.where(mask, values, float("nan"))
    medians = pool.nanmedian(dim=1, keepdim=True).values
    return (pool - medians).abs().nanmedian(dim=1, keepdim=True).values


def _filter_events(fitness, orders, generator, epsilon=None, dynamic_epsilon=False):
    """Run every selection event together, one case position at a time."""
    n_individuals = fitness.shape[0]
    n_events, n_cases = orders.shape

    mask = torch.ones(
        (n_events, n_individuals), dtype=torch.bool, device=fitness.device
    )

    for position in range(n_cases):
        cases = orders[:, position]
        values = fitness.index_select(1, cases).t()
        values = torch.where(mask, values, float("-inf"))
        best = values.max(dim=1, keepdim=True).values

        if dynamic_epsilon:
            tolerance = _pool_mad(values, mask)
        elif epsilon is None:
            tolerance = 0.0
        elif isinstance(epsilon, torch.Tensor):
            tolerance = epsilon.index_select(0, cases).unsqueeze(1)
        else:
            tolerance = epsilon

        survivors = mask & (values >= best - tolerance)
        still_open = mask.sum(dim=1, keepdim=True) > 1
        mask = torch.where(still_open, survivors, mask)

    return _random_from_mask(mask, fitness, generator)


def _select_elites(fitness: torch.Tensor, elitism: int) -> torch.Tensor:
    return torch.topk(fitness.sum(dim=1), elitism).indices


def _with_elites(fitness, elitism, num_selected, body):
    if elitism <= 0:
        return body(num_selected)
    elites = _select_elites(fitness, elitism)
    if num_selected == elitism:
        return elites
    return torch.cat([elites, body(num_selected - elitism)])


def _empty(fitness: torch.Tensor) -> torch.Tensor:
    return torch.empty(0, dtype=torch.long, device=fitness.device)


def torch_compute_mad_epsilon(fitness_matrix: torch.Tensor) -> torch.Tensor:
    """Median absolute deviation of each case, floored at MIN_EPSILON."""
    fitness = sanitize(fitness_matrix)
    medians = fitness.median(dim=0, keepdim=True).values
    mad = (fitness - medians).abs().median(dim=0).values
    return mad.clamp_min(MIN_EPSILON)


def torch_lexicase_selection(
    fitness_matrix: torch.Tensor,
    num_selected: int,
    seed: Optional[int] = None,
    elitism: int = 0,
    case_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Torch lexicase selection.

    Reference:
        Helmuth, T., Spector, L., and Matheson, J. (2015). Solving
        Uncompromising Problems with Lexicase Selection. IEEE Transactions on
        Evolutionary Computation 19(5), 630-643.

    Returns a long tensor of indices on the input tensor's device.
    """
    if num_selected == 0:
        return _empty(fitness_matrix)

    fitness = sanitize(fitness_matrix)
    generator = _generator(fitness, seed)
    n_cases = fitness.shape[1]
    weights = to_device(case_weights, fitness)

    def body(count):
        orders = _case_orders(count, n_cases, fitness, generator, weights)
        return _filter_events(fitness, orders, generator)

    return _with_elites(fitness, elitism, num_selected, body)


def torch_epsilon_lexicase_selection(
    fitness_matrix: torch.Tensor,
    num_selected: int,
    epsilon,
    seed: Optional[int] = None,
    elitism: int = 0,
    mode: str = "semi-dynamic",
    case_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Torch epsilon lexicase selection, in all three modes.

    Reference:
        La Cava, W., Helmuth, T., Spector, L., and Moore, J. H. (2019). A
        probabilistic and multi-objective analysis of lexicase selection and
        epsilon-lexicase selection. Evolutionary Computation 27(3), 377-402.
        Algorithms 2, 3, and 4.
    """
    if num_selected == 0:
        return _empty(fitness_matrix)

    fitness = sanitize(fitness_matrix)
    generator = _generator(fitness, seed)
    n_cases = fitness.shape[1]
    weights = to_device(case_weights, fitness)

    tolerance = to_device(epsilon, fitness)
    if isinstance(tolerance, torch.Tensor):
        tolerance = tolerance.expand(n_cases).contiguous()

    filter_matrix = fitness
    if mode == "static":
        case_best = fitness.max(dim=0, keepdim=True).values
        offset = tolerance[None, :] if isinstance(tolerance, torch.Tensor) else tolerance
        filter_matrix = (fitness >= case_best - offset).to(fitness.dtype)
        tolerance = None
    elif mode == "dynamic":
        tolerance = None
    elif mode != "semi-dynamic":
        raise ValueError(f"Unknown epsilon mode {mode!r}")

    def body(count):
        orders = _case_orders(count, n_cases, fitness, generator, weights)
        return _filter_events(
            filter_matrix, orders, generator, tolerance, mode == "dynamic"
        )

    return _with_elites(fitness, elitism, num_selected, body)


def torch_downsample_lexicase_selection(
    fitness_matrix: torch.Tensor,
    num_selected: int,
    downsample_size: int,
    seed: Optional[int] = None,
    elitism: int = 0,
) -> torch.Tensor:
    """
    Torch downsampled lexicase selection.

    A random permutation truncated to `downsample_size` is both the random
    subset and its order, so one draw does the whole job.

    Reference:
        Hernandez, J. G., Lalejini, A., Dolson, E., and Ofria, C. (2019).
        Random subsampling improves performance in lexicase selection.
        GECCO '19 Companion, pp. 2028-2031.
    """
    if num_selected == 0:
        return _empty(fitness_matrix)

    fitness = sanitize(fitness_matrix)
    generator = _generator(fitness, seed)
    n_cases = fitness.shape[1]
    size = min(downsample_size, n_cases)

    def body(count):
        orders = _case_orders(count, n_cases, fitness, generator)[:, :size]
        return _filter_events(fitness, orders, generator)

    return _with_elites(fitness, elitism, num_selected, body)


def _farthest_first_traversal(distances, downsample_size, fitness, generator):
    n_cases = distances.shape[0]
    if downsample_size >= n_cases:
        return torch.arange(n_cases, device=distances.device)

    noise = _random((n_cases,), fitness, generator)
    first = noise.argmax()
    picks = [first]
    chosen = torch.zeros(n_cases, dtype=torch.bool, device=distances.device)
    chosen = chosen.index_fill(0, first.view(1), True)
    min_distance = distances.index_select(0, first.view(1)).squeeze(0)

    for _ in range(1, downsample_size):
        noise = _random((n_cases,), fitness, generator)
        scores = torch.where(chosen, float("-inf"), min_distance)
        ties = scores == scores.max()
        pick = torch.where(ties, noise, -1.0).argmax()
        picks.append(pick)
        chosen = chosen.index_fill(0, pick.view(1), True)
        min_distance = torch.minimum(
            min_distance, distances.index_select(0, pick.view(1)).squeeze(0)
        )

    return torch.stack(picks)


def torch_informed_downsample_lexicase_selection(
    fitness_matrix: torch.Tensor,
    num_selected: int,
    downsample_size: int,
    seed: Optional[int] = None,
    sample_rate: float = 0.01,
    threshold=None,
    elitism: int = 0,
) -> torch.Tensor:
    """
    Torch informed downsampled lexicase selection.

    The case subset is chosen once per call by farthest first traversal over
    Hamming distances between case solve patterns, then reused for every
    selection event, matching the other backends.

    `threshold` is the pass/fail cutoff. The NumPy and JAX backends detect a
    pass/fail matrix and set it for you; this one never does, because reading the
    values would synchronize with the host and this backend exists not to. For
    0/1 rewards pass `threshold=0.5`. With `threshold=None` you get a per-case
    median split, which is a heuristic and not the rule in the paper.

    Reference:
        Boldi, R., Briesch, M., Sobania, D., Lalejini, A., Helmuth, T.,
        Rothlauf, F., Ofria, C., and Spector, L. (2024). Informed Down-Sampled
        Lexicase Selection: Identifying Productive Training Cases for Efficient
        Problem Solving. Evolutionary Computation 32(4), 307-337.
    """
    if num_selected == 0:
        return _empty(fitness_matrix)

    fitness = sanitize(fitness_matrix)
    generator = _generator(fitness, seed)
    n_individuals, n_cases = fitness.shape
    size = min(downsample_size, n_cases)
    n_sampled = max(1, int(n_individuals * sample_rate))

    order = torch.argsort(_random((n_individuals,), fitness, generator))
    sampled = fitness.index_select(0, order[:n_sampled])

    if threshold is None:
        cutoff = sampled.median(dim=0, keepdim=True).values
    else:
        cutoff = to_device(threshold, fitness)
        if isinstance(cutoff, torch.Tensor):
            cutoff = cutoff.expand(n_cases)[None, :]
    solved = (sampled > cutoff).to(fitness.dtype)

    mismatch = solved.t() @ (1.0 - solved)
    cases = _farthest_first_traversal(mismatch + mismatch.t(), size, fitness, generator)
    submatrix = fitness.index_select(1, cases)

    def body(count):
        orders = _case_orders(count, submatrix.shape[1], submatrix, generator)
        return _filter_events(submatrix, orders, generator)

    return _with_elites(fitness, elitism, num_selected, body)


def torch_batch_lexicase_selection(
    fitness_matrix: torch.Tensor,
    num_selected: int,
    batch_size: int,
    seed: Optional[int] = None,
    threshold: Optional[float] = None,
    elitism: int = 0,
) -> torch.Tensor:
    """
    Torch batch lexicase selection.

    Reference:
        Aenugu, S. and Spector, L. (2019). Lexicase Selection in Learning
        Classifier Systems. GECCO '19, pp. 356-364. Algorithm 2.
    """
    if num_selected == 0:
        return _empty(fitness_matrix)

    fitness = sanitize(fitness_matrix)
    generator = _generator(fitness, seed)
    n_individuals, n_cases = fitness.shape

    def body(count):
        orders = _case_orders(count, n_cases, fitness, generator)
        mask = torch.ones(
            (count, n_individuals), dtype=torch.bool, device=fitness.device
        )

        for start in range(0, n_cases, batch_size):
            cases = orders[:, start : start + batch_size]
            scores = fitness[:, cases].mean(dim=2).t()
            scores = torch.where(mask, scores, float("-inf"))

            if threshold is None:
                survivors = mask & (scores >= scores.max(dim=1, keepdim=True).values)
            else:
                survivors = mask & (scores > threshold)
                survivors = torch.where(
                    survivors.any(dim=1, keepdim=True), survivors, mask
                )

            still_open = mask.sum(dim=1, keepdim=True) > 1
            mask = torch.where(still_open, survivors, mask)

        return _random_from_mask(mask, fitness, generator)

    return _with_elites(fitness, elitism, num_selected, body)


def torch_cohort_lexicase_selection(
    fitness_matrix: torch.Tensor,
    num_selected: int,
    num_cohorts: int,
    seed: Optional[int] = None,
    elitism: int = 0,
) -> torch.Tensor:
    """
    Torch cohort lexicase selection.

    Reference:
        Hernandez, J. G., Lalejini, A., Dolson, E., and Ofria, C. (2019).
        Random subsampling improves performance in lexicase selection.
        GECCO '19 Companion, pp. 2028-2031. Section 4.
    """
    if num_selected == 0:
        return _empty(fitness_matrix)

    fitness = sanitize(fitness_matrix)
    generator = _generator(fitness, seed)
    n_individuals, n_cases = fitness.shape

    if num_cohorts > n_individuals or num_cohorts > n_cases:
        raise ValueError("Number of cohorts cannot exceed individuals or cases")

    def body(count):
        individual_order = torch.argsort(_random((n_individuals,), fitness, generator))
        case_order = torch.argsort(_random((n_cases,), fitness, generator))
        individual_bounds = split_bounds(n_individuals, num_cohorts)
        case_bounds = split_bounds(n_cases, num_cohorts)
        counts = split_counts(count, num_cohorts)

        parts = []
        for cohort in range(num_cohorts):
            if counts[cohort] == 0:
                continue
            start, size = individual_bounds[cohort]
            members = individual_order[start : start + size]
            start, size = case_bounds[cohort]
            cases = case_order[start : start + size]
            submatrix = fitness.index_select(0, members).index_select(1, cases)
            orders = _case_orders(counts[cohort], len(cases), submatrix, generator)
            local = _filter_events(submatrix, orders, generator)
            parts.append(members.index_select(0, local))
        return torch.cat(parts)

    return _with_elites(fitness, elitism, num_selected, body)


def split_bounds(total, parts):
    """Offsets and sizes matching numpy.array_split."""
    base, extra = divmod(total, parts)
    bounds = []
    offset = 0
    for index in range(parts):
        size = base + (1 if index < extra else 0)
        bounds.append((offset, size))
        offset += size
    return bounds


def split_counts(total, parts):
    base, extra = divmod(total, parts)
    return [base + (1 if index < extra else 0) for index in range(parts)]


def torch_dalex_selection(
    fitness_matrix: torch.Tensor,
    num_selected: int,
    seed: Optional[int] = None,
    particularity_pressure: float = 20.0,
    relaxed: bool = False,
    elitism: int = 0,
) -> torch.Tensor:
    """
    Torch diversely aggregated lexicase selection (DALex).

    One softmax and one matrix multiply, which makes this the cheapest variant
    on an accelerator by a wide margin.

    Reference:
        Ni, A., Ding, L., and Spector, L. (2024). DALex: Lexicase-like Selection
        via Diverse Aggregation. EuroGP 2024, LNCS 14631, pp. 90-107. Algorithm 1.
    """
    if num_selected == 0:
        return _empty(fitness_matrix)

    fitness = sanitize(fitness_matrix)
    generator = _generator(fitness, seed)
    n_cases = fitness.shape[1]

    scores_matrix = fitness
    if relaxed:
        spread = scores_matrix.std(dim=0, keepdim=True)
        spread = torch.where(spread > 0, spread, torch.ones_like(spread))
        scores_matrix = (scores_matrix - scores_matrix.mean(dim=0, keepdim=True)) / spread

    def body(count):
        importance = torch.randn(
            (count, n_cases),
            generator=generator,
            device=fitness.device,
            dtype=fitness.dtype,
        )
        weights = torch.softmax(importance * particularity_pressure, dim=1)
        return (scores_matrix @ weights.t()).argmax(dim=0)

    return _with_elites(fitness, elitism, num_selected, body)
