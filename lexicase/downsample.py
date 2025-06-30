"""
Downsampled lexicase selection implementation.
"""

from .backend import get_lib
from .utils import validate_fitness_matrix, validate_selection_params, shuffle_cases


def downsample_lexicase_selection(fitness_matrix, num_selected, downsample_size, seed=None):
    """Perform downsampled lexicase selection on a population.
    
    Downsampled lexicase selection randomly selects a subset of test cases
    for each individual selection, which can increase diversity and reduce
    computational cost.
    
    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases) containing
                       fitness values. Higher values indicate better performance.
        num_selected: Number of individuals to select
        downsample_size: Number of test cases to randomly sample for each selection
        seed: Random seed for reproducibility
        
    Returns:
        Array of selected individual indices
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    fitness_matrix = validate_fitness_matrix(fitness_matrix)
    validate_selection_params(num_selected, seed)
    
    if downsample_size <= 0:
        raise ValueError("Downsample size must be positive")
    
    if num_selected == 0:
        xp, _ = get_lib()
        return xp.array([], dtype=int)
    
    xp, rng = get_lib(seed)
    
    n_individuals, n_cases = fitness_matrix.shape
    actual_downsample_size = min(downsample_size, n_cases)
    selected = []
    
    # Perform selection
    for _ in range(num_selected):
        # Randomly sample test cases for this selection
        if hasattr(rng, 'choice'):  # NumPy
            sampled_cases = rng.choice(
                n_cases, 
                size=actual_downsample_size, 
                replace=False
            )
        else:  # JAX
            import jax
            rng, _rng = jax.random.split(rng)
            sampled_cases = jax.random.choice(
                rng, 
                n_cases, 
                shape=(actual_downsample_size,), 
                replace=False
            )
        
        # Create submatrix with only sampled cases
        submatrix = fitness_matrix[:, sampled_cases]
        
        # Perform lexicase selection on the submatrix
        candidates = xp.arange(n_individuals)
        case_order = shuffle_cases(actual_downsample_size, rng)
        
        # Filter candidates case by case
        for case_idx in case_order:
            if len(candidates) <= 1:
                break
                
            case_fitness = submatrix[candidates, case_idx]
            max_fitness = xp.max(case_fitness)
            
            # Keep only individuals with maximum fitness on this case
            best_mask = case_fitness == max_fitness
            candidates = candidates[best_mask]
        
        # Randomly select one from remaining candidates
        if len(candidates) == 1:
            selected.append(int(candidates[0]))
        else:
            # Multiple candidates remain - select randomly
            if hasattr(rng, 'choice'):  # NumPy
                chosen_idx = rng.choice(len(candidates))
            else:  # JAX
                import jax
                chosen_idx = jax.random.choice(rng, len(candidates))
            selected.append(int(candidates[chosen_idx]))
    
    return xp.array(selected, dtype=int) 