"""
Tournament selection implementation for comparison with lexicase selection.
"""

import numpy as np


def tournament_selection(fitness_matrix, num_selected, tournament_size=3, seed=None):
    """
    Tournament selection algorithm.
    
    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases) containing
                       fitness values. Higher values indicate better performance.
        num_selected: Number of individuals to select
        tournament_size: Number of individuals in each tournament (default: 3)
        seed: Random seed for reproducibility
        
    Returns:
        Array of selected individual indices
        
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
    
    if seed is not None and not isinstance(seed, int):
        raise ValueError("Seed must be an integer")
    
    n_individuals = fitness_matrix.shape[0]
    rng = np.random.default_rng(seed)
    
    # Calculate aggregate fitness (sum across all test cases)
    aggregate_fitness = np.sum(fitness_matrix, axis=1)
    
    selected = []
    
    for _ in range(num_selected):
        # Select tournament participants
        tournament_indices = rng.choice(n_individuals, size=tournament_size, replace=True)
        tournament_fitness = aggregate_fitness[tournament_indices]
        
        # Find winner (highest aggregate fitness)
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(winner_idx)
    
    return np.array(selected, dtype=np.int32)