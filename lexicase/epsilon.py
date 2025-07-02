"""
Epsilon lexicase selection implementation.
"""

from .backend import get_lib
from .utils import validate_fitness_matrix, validate_selection_params, compute_mad_epsilon


def epsilon_lexicase_selection(fitness_matrix, num_selected, epsilon=None, seed=None):
    """Epsilon lexicase selection algorithm.
    
    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases) containing
                       fitness values. Higher values indicate better performance.
        num_selected: Number of individuals to select
        epsilon: Tolerance value for "equal" performance. If None (default),
                uses Median Absolute Deviation (MAD) for each case.
                Can be a scalar (same epsilon for all cases) or array-like
                (different epsilon per case).
        seed: Random seed for reproducibility
        
    Returns:
        Array of selected individual indices
        
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
    
    # Determine epsilon values for each case
    if epsilon is None:
        # Use MAD as default epsilon for each case
        epsilon_values = compute_mad_epsilon(fitness_matrix)
    else:
        # Validate provided epsilon
        epsilon_array = xp.asarray(epsilon)
        
        if epsilon_array.ndim == 0:
            # Scalar epsilon - broadcast to all cases
            if epsilon < 0:
                raise ValueError("Epsilon must be non-negative")
            epsilon_values = xp.full(n_cases, epsilon_array)
        else:
            # Array epsilon - validate length and values
            if len(epsilon_array) != n_cases:
                raise ValueError(f"Epsilon array length ({len(epsilon_array)}) must match number of cases ({n_cases})")
            if xp.any(epsilon_array < 0):
                raise ValueError("All epsilon values must be non-negative")
            epsilon_values = epsilon_array
    
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
            case_epsilon = epsilon_values[case_idx]
            
            # Keep only individuals within epsilon of the best performance
            best_mask = case_fitness >= (max_fitness - case_epsilon)
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