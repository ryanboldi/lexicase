"""
Pure NumPy implementations of lexicase selection algorithms.

These functions are optimized for NumPy arrays and avoid JAX dependencies
for users who don't need GPU acceleration.
"""

import numpy as np


def numpy_lexicase_selection(fitness_matrix, num_selected, rng):
    """
    NumPy-based lexicase selection implementation.
    
    Args:
        fitness_matrix: NumPy array of shape (n_individuals, n_cases)
                       Higher values indicate better performance.
        num_selected: Number of individuals to select (int)
        rng: NumPy random number generator (from np.random.default_rng())
        
    Returns:
        NumPy array of selected individual indices
    """
    if num_selected == 0:
        return np.array([], dtype=int)
    
    n_individuals, n_cases = fitness_matrix.shape
    selected = []
    
    # Perform selection
    for _ in range(num_selected):
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


def numpy_epsilon_lexicase_selection(fitness_matrix, num_selected, epsilon, rng):
    """
    NumPy-based epsilon lexicase selection implementation.
    
    Args:
        fitness_matrix: NumPy array of shape (n_individuals, n_cases)
        num_selected: Number of individuals to select
        epsilon: Tolerance value(s). Can be scalar or array of length n_cases
        rng: NumPy random number generator
        
    Returns:
        NumPy array of selected individual indices
    """
    if num_selected == 0:
        return np.array([], dtype=int)
    
    n_individuals, n_cases = fitness_matrix.shape
    
    # Handle epsilon - ensure it's the right shape
    epsilon_values = np.broadcast_to(epsilon, (n_cases,))
    
    selected = []
    
    # Perform selection
    for _ in range(num_selected):
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


def numpy_epsilon_lexicase_selection_with_mad(fitness_matrix, num_selected, rng):
    """
    NumPy epsilon lexicase selection using MAD-based adaptive epsilon.
    
    Args:
        fitness_matrix: NumPy array of shape (n_individuals, n_cases)
        num_selected: Number of individuals to select
        rng: NumPy random number generator
        
    Returns:
        NumPy array of selected individual indices
    """
    # Compute MAD-based epsilon values
    epsilon_values = numpy_compute_mad_epsilon(fitness_matrix)
    
    # Use epsilon lexicase with computed epsilon
    return numpy_epsilon_lexicase_selection(fitness_matrix, num_selected, epsilon_values, rng)


def numpy_downsample_lexicase_selection(fitness_matrix, num_selected, downsample_size, rng):
    """
    NumPy-based downsampled lexicase selection implementation.
    
    Args:
        fitness_matrix: NumPy array of shape (n_individuals, n_cases)
        num_selected: Number of individuals to select
        downsample_size: Number of test cases to randomly sample for each selection
        rng: NumPy random number generator
        
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
    
    # Perform selection
    for _ in range(num_selected):
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