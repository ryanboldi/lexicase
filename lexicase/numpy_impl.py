"""
Pure NumPy implementations of lexicase selection algorithms.

These functions are optimized for NumPy arrays and avoid JAX dependencies
for users who don't need GPU acceleration.
"""

import numpy as np


def numpy_lexicase_selection(fitness_matrix, num_selected, rng, elitism=0):
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
        return np.array([], dtype=int)
    
    n_individuals, n_cases = fitness_matrix.shape
    selected = []
    
    # Handle elitism: select best individuals by total fitness
    if elitism > 0:
        # Calculate total fitness for each individual
        total_fitness = np.sum(fitness_matrix, axis=1)
        # Get indices of top performers
        elite_indices = np.argsort(total_fitness)[-elitism:]
        # Add elite individuals to selection
        selected.extend(elite_indices.tolist())
    
    # Perform regular lexicase selection for remaining slots
    for _ in range(num_selected - elitism):
        # Shuffle the order of test cases
        case_order = rng.permutation(n_cases)
        
        # Start with all individuals as candidates
        candidates = np.arange(n_individuals)
        
        # Filter candidates case by case
        for case_idx in case_order:
            if len(candidates) <= 1:
                break
                
            case_fitness = fitness_matrix[candidates, case_idx]
            max_fitness = np.max(case_fitness)
            
            # Keep only individuals with maximum fitness on this case
            best_mask = case_fitness == max_fitness
            candidates = candidates[best_mask]
        
        # Randomly select one from remaining candidates
        if len(candidates) == 1:
            selected.append(int(candidates[0]))
        else:
            # Multiple candidates remain - select randomly
            chosen_idx = rng.choice(len(candidates))
            selected.append(int(candidates[chosen_idx]))

    return np.array(selected, dtype=int)


def numpy_epsilon_lexicase_selection(fitness_matrix, num_selected, epsilon, rng, elitism=0):
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
        return np.array([], dtype=int)
    
    n_individuals, n_cases = fitness_matrix.shape
    
    # Handle epsilon - ensure it's the right shape
    epsilon_values = np.broadcast_to(epsilon, (n_cases,))
    
    selected = []
    
    # Handle elitism: select best individuals by total fitness
    if elitism > 0:
        # Calculate total fitness for each individual
        total_fitness = np.sum(fitness_matrix, axis=1)
        # Get indices of top performers
        elite_indices = np.argsort(total_fitness)[-elitism:]
        # Add elite individuals to selection
        selected.extend(elite_indices.tolist())
    
    # Perform selection for remaining slots
    for _ in range(num_selected - elitism):
        # Shuffle the order of test cases
        case_order = rng.permutation(n_cases)
        
        # Start with all individuals as candidates
        candidates = np.arange(n_individuals)
        
        # Filter candidates case by case
        for case_idx in case_order:
            if len(candidates) <= 1:
                break
                
            case_fitness = fitness_matrix[candidates, case_idx]
            max_fitness = np.max(case_fitness)
            case_epsilon = epsilon_values[case_idx]
            
            # Keep individuals within epsilon of the best performance
            best_mask = case_fitness >= (max_fitness - case_epsilon)
            candidates = candidates[best_mask]
        
        # Randomly select one from remaining candidates
        if len(candidates) == 1:
            selected.append(int(candidates[0]))
        else:
            # Multiple candidates remain - select randomly
            chosen_idx = rng.choice(len(candidates))
            selected.append(int(candidates[chosen_idx]))

    return np.array(selected, dtype=int)


def numpy_compute_mad_epsilon(fitness_matrix):
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
    min_epsilon = 1e-10
    mad_values = np.maximum(mad_values, min_epsilon)
    
    return mad_values


def numpy_epsilon_lexicase_selection_with_mad(fitness_matrix, num_selected, rng, elitism=0):
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
    return numpy_epsilon_lexicase_selection(fitness_matrix, num_selected, epsilon_values, rng, elitism)


def numpy_downsample_lexicase_selection(fitness_matrix, num_selected, downsample_size, rng, elitism=0):
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
        return np.array([], dtype=int)
    
    if downsample_size <= 0:
        raise ValueError("Downsample size must be positive")
    
    n_individuals, n_cases = fitness_matrix.shape
    actual_downsample_size = min(downsample_size, n_cases)
    selected = []
    
    # Handle elitism: select best individuals by total fitness
    if elitism > 0:
        # Calculate total fitness for each individual
        total_fitness = np.sum(fitness_matrix, axis=1)
        # Get indices of top performers
        elite_indices = np.argsort(total_fitness)[-elitism:]
        # Add elite individuals to selection
        selected.extend(elite_indices.tolist())
    
    # Perform selection for remaining slots
    for _ in range(num_selected - elitism):
        # Randomly sample test cases for this selection
        sampled_cases = rng.choice(
            n_cases, 
            size=actual_downsample_size, 
            replace=False
        )
        
        # Create submatrix with only sampled cases
        submatrix = fitness_matrix[:, sampled_cases]
        
        # Shuffle case order for the submatrix
        case_order = rng.permutation(actual_downsample_size)
        
        # Perform lexicase selection on the submatrix
        candidates = np.arange(n_individuals)
        
        # Filter candidates case by case
        for case_idx in case_order:
            if len(candidates) <= 1:
                break
                
            case_fitness = submatrix[candidates, case_idx]
            max_fitness = np.max(case_fitness)
            
            # Keep only individuals with maximum fitness on this case
            best_mask = case_fitness == max_fitness
            candidates = candidates[best_mask]
        
        # Randomly select one from remaining candidates
        if len(candidates) == 1:
            selected.append(int(candidates[0]))
        else:
            # Multiple candidates remain - select randomly
            chosen_idx = rng.choice(len(candidates))
            selected.append(int(candidates[chosen_idx]))
    
    return np.array(selected, dtype=int)


def _compute_case_distances(fitness_matrix, sample_indices, threshold=None):
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
        solve_matrix = sampled_fitness > threshold[None, :]
    
    # Compute Hamming distances between cases
    distances = np.zeros((n_cases, n_cases))
    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            # Hamming distance: count differences in solve patterns
            distance = np.sum(solve_matrix[:, i] != solve_matrix[:, j])
            distances[i, j] = distance
            distances[j, i] = distance
    
    return distances


def _farthest_first_traversal(distances, downsample_size, rng):
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
        return np.arange(n_cases)
    
    selected = []
    remaining = list(range(n_cases))
    
    # Randomly select first case
    first_idx = rng.choice(remaining)
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
        min_distances = np.array(min_distances)
        max_min_dist = np.max(min_distances)
        
        # Handle ties randomly
        candidates = [remaining[i] for i in range(len(remaining)) 
                     if min_distances[i] == max_min_dist]
        
        if candidates:
            chosen = rng.choice(candidates)
            selected.append(chosen)
            remaining.remove(chosen)
        else:
            # If all distances are 0, randomly select from remaining
            chosen = rng.choice(remaining)
            selected.append(chosen)
            remaining.remove(chosen)
    
    return np.array(selected)


def numpy_informed_downsample_lexicase_selection(
    fitness_matrix, num_selected, downsample_size, rng, 
    sample_rate=0.01, threshold=None, elitism=0
):
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
        return np.array([], dtype=int)
    
    if downsample_size <= 0:
        raise ValueError("Downsample size must be positive")
    
    n_individuals, n_cases = fitness_matrix.shape
    actual_downsample_size = min(downsample_size, n_cases)
    
    selected = []
    
    # Handle elitism: select best individuals by total fitness
    if elitism > 0:
        # Calculate total fitness for each individual
        total_fitness = np.sum(fitness_matrix, axis=1)
        # Get indices of top performers
        elite_indices = np.argsort(total_fitness)[-elitism:]
        # Add elite individuals to selection
        selected.extend(elite_indices.tolist())
    
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
    for _ in range(num_selected - elitism):
        # Shuffle case order for the submatrix
        case_order = rng.permutation(actual_downsample_size)
        
        # Perform lexicase selection on the submatrix
        candidates = np.arange(n_individuals)
        
        # Filter candidates case by case
        for case_idx in case_order:
            if len(candidates) <= 1:
                break
                
            case_fitness = submatrix[candidates, case_idx]
            max_fitness = np.max(case_fitness)
            
            # Keep only individuals with maximum fitness on this case
            best_mask = case_fitness == max_fitness
            candidates = candidates[best_mask]
        
        # Randomly select one from remaining candidates
        if len(candidates) == 1:
            selected.append(int(candidates[0]))
        else:
            # Multiple candidates remain - select randomly
            chosen_idx = rng.choice(len(candidates))
            selected.append(int(candidates[chosen_idx]))
    
    return np.array(selected, dtype=int)