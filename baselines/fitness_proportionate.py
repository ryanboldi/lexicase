"""Fitness proportionate (roulette wheel) selection, for comparison."""

import numpy as np


def fitness_proportionate_selection(fitness_matrix, num_selected, seed=None):
    """
    Roulette wheel selection on aggregate fitness.

    Aggregate fitness is shifted so the worst individual sits at zero, which
    keeps the method defined for fitness matrices containing negative values.
    If every individual aggregates to the same value, selection is uniform.

    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases). Higher is better.
        num_selected: Number of individuals to select
        seed: Random seed for reproducibility

    Returns:
        NumPy array of selected individual indices

    Raises:
        ValueError: If inputs are invalid
    """
    fitness_matrix = np.asarray(fitness_matrix)

    if fitness_matrix.ndim != 2:
        raise ValueError("Fitness matrix must be 2-dimensional")
    if num_selected < 0:
        raise ValueError("Number of selected individuals must be non-negative")

    rng = np.random.default_rng(seed)
    aggregate = np.sum(fitness_matrix, axis=1)
    weights = aggregate - np.min(aggregate)
    total = np.sum(weights)

    if total <= 0:
        probabilities = np.full(len(aggregate), 1.0 / len(aggregate))
    else:
        probabilities = weights / total

    return rng.choice(len(aggregate), size=num_selected, replace=True, p=probabilities)
