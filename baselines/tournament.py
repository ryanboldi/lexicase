"""Tournament selection, for comparison against lexicase selection."""

import numpy as np


def tournament_selection(fitness_matrix, num_selected, tournament_size=3, seed=None):
    """
    Tournament selection on aggregate fitness.

    Each selection event draws `tournament_size` individuals with replacement
    and keeps the one with the highest total fitness across all cases.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases). Higher is better.
        num_selected: Number of individuals to select
        tournament_size: Number of individuals per tournament
        seed: Random seed for reproducibility

    Returns:
        NumPy array of selected individual indices

    Raises:
        ValueError: If inputs are invalid
    """
    fitness_matrix = np.asarray(fitness_matrix)

    if fitness_matrix.ndim != 2:
        raise ValueError("Fitness matrix must be 2-dimensional")
    if fitness_matrix.shape[0] == 0:
        raise ValueError("Fitness matrix must have at least one individual")
    if fitness_matrix.shape[1] == 0:
        raise ValueError("Fitness matrix must have at least one test case")
    if num_selected < 0:
        raise ValueError("Number of selected individuals must be non-negative")
    if tournament_size <= 0:
        raise ValueError("Tournament size must be positive")
    if seed is not None and not isinstance(seed, (int, np.integer)):
        raise ValueError("Seed must be an integer")

    n_individuals = fitness_matrix.shape[0]
    rng = np.random.default_rng(seed)
    aggregate = np.sum(fitness_matrix, axis=1)

    entrants = rng.choice(n_individuals, size=(num_selected, tournament_size), replace=True)
    winners = entrants[np.arange(num_selected), np.argmax(aggregate[entrants], axis=1)]
    return winners.astype(np.intp)
