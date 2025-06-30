"""
Base lexicase selection implementation.
"""

from .backend import get_lib
from .utils import validate_fitness_matrix, validate_selection_params


def lexicase_selection(fitness_matrix, num_selected, seed=None):
    """
    Lexicase selection algorithm
    
    Args:
        fitness_matrix (array): fitnesses of the population. This should be an
            array of shape (num_individuals, num_objectives). Higher values 
            indicate better performance.
        num_selected (int): number of individuals to select.
        seed: Random seed for reproducibility
            
    Returns:
        Array of selected individual indices (shape (num_selected,))
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    fitness_matrix = validate_fitness_matrix(fitness_matrix)
    validate_selection_params(num_selected, seed)
    
    if num_selected == 0:
        xp, _ = get_lib()
        return xp.array([], dtype=int)

    xp, rng = get_lib(seed)
    
    n_individuals, n_cases = fitness_matrix.shape
    selected = []
    
    # Perform selection
    for _ in range(num_selected):
        # Shuffle the order of test cases
        case_order = xp.arange(n_cases)
        if hasattr(rng, 'permutation'):  # NumPy
            case_order = rng.permutation(case_order)
        else:  # JAX
            import jax
            rng, _rng = jax.random.split(rng)
            case_order = jax.random.permutation(_rng, case_order)
            
        
        # Start with all individuals as candidates
        candidates = xp.arange(n_individuals)
        
        # Filter candidates case by case
        for case_idx in case_order:
            if len(candidates) <= 1:
                break
                
            case_fitness = fitness_matrix[candidates, case_idx]
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