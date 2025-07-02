"""
Utility functions for lexicase selection.
"""

from .backend import get_lib


def validate_fitness_matrix(fitness_matrix):
    """Validate fitness matrix format and content.
    
    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases)
        
    Raises:
        ValueError: If matrix format is invalid
    """
    xp, _ = get_lib()
    fitness_matrix = xp.asarray(fitness_matrix)
    
    if fitness_matrix.ndim != 2:
        raise ValueError("Fitness matrix must be 2-dimensional")
    
    if fitness_matrix.shape[0] == 0:
        raise ValueError("Fitness matrix must have at least one individual")
    
    if fitness_matrix.shape[1] == 0:
        raise ValueError("Fitness matrix must have at least one test case")
    
    return fitness_matrix


def validate_selection_params(num_selected, seed=None):
    """Validate selection parameters.
    
    Args:
        num_selected: Number of individuals to select
        seed: Random seed
        
    Raises:
        ValueError: If parameters are invalid
    """
    if num_selected < 0:
        raise ValueError("Number of selected individuals must be non-negative")
    
    if seed is not None and not isinstance(seed, int):
        raise ValueError("Seed must be an integer")


def shuffle_cases(num_cases, rng):
    """Shuffle test case indices.
    
    Args:
        num_cases: Number of test cases
        rng: Random number generator
        
    Returns:
        Array of shuffled case indices
    """
    xp, _ = get_lib()
    
    if hasattr(rng, 'permutation'):  # NumPy
        return rng.permutation(num_cases)
    else:  # JAX
        import jax
        return jax.random.permutation(rng, num_cases)


def compute_case_variance(fitness_matrix):
    """Compute variance for each test case across individuals.
    
    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases)
        
    Returns:
        Array of variances for each test case
    """
    xp, _ = get_lib()
    return xp.var(fitness_matrix, axis=0)


def select_informative_cases(fitness_matrix, downsample_size, rng):
    """Select informative test cases based on variance.
    
    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases)
        downsample_size: Number of cases to select
        rng: Random number generator
        
    Returns:
        Array of selected case indices
    """
    xp, _ = get_lib()
    
    variances = compute_case_variance(fitness_matrix)
    num_cases = fitness_matrix.shape[1]
    
    # If all variances are zero or very small, fall back to uniform sampling
    if xp.all(variances < 1e-10):
        case_indices = xp.arange(num_cases)
        if hasattr(rng, 'choice'):  # NumPy
            return rng.choice(case_indices, size=min(downsample_size, num_cases), replace=False)
        else:  # JAX
            import jax
            return jax.random.choice(rng, case_indices, shape=(min(downsample_size, num_cases),), replace=False)
    
    # Select cases with probability proportional to variance
    probabilities = variances / xp.sum(variances)
    
    if hasattr(rng, 'choice'):  # NumPy
        return rng.choice(
            num_cases, 
            size=min(downsample_size, num_cases), 
            replace=False, 
            p=probabilities
        )
    else:  # JAX
        import jax
        return jax.random.choice(
            rng, 
            num_cases, 
            shape=(min(downsample_size, num_cases),), 
            replace=False, 
            p=probabilities
        )


def compute_mad_epsilon(fitness_matrix):
    """Compute Median Absolute Deviation (MAD) for each test case.
    
    MAD is calculated as the median of absolute deviations from the median
    for each test case across all individuals.
    
    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases)
        
    Returns:
        Array of MAD values for each test case
    """
    xp, _ = get_lib()
    
    # Calculate median for each case (column) - more robust than mean
    case_medians = xp.median(fitness_matrix, axis=0)
    
    # Calculate absolute deviations from median for each case
    abs_deviations = xp.abs(fitness_matrix - case_medians[None, :])
    
    # Calculate median of absolute deviations for each case
    mad_values = xp.median(abs_deviations, axis=0)
    
    # Handle case where MAD is 0 (all values identical) by using a small default
    min_epsilon = 1e-10
    mad_values = xp.maximum(mad_values, min_epsilon)
    
    return mad_values

