"""
Pure JAX implementations of lexicase selection algorithms.

These functions are designed to be JIT-compilable and work efficiently 
with JAX arrays on CPU/GPU/TPU.
"""

import jax
import jax.numpy as jnp
from jax import lax


def jax_lexicase_selection_impl(fitness_matrix, num_selected, key):
    """
    JAX-based lexicase selection implementation.
    
    This function is JIT-compiled for maximum performance.
    Assumes num_selected > 0.
    
    Args:
        fitness_matrix: JAX array of shape (n_individuals, n_cases)
                       Higher values indicate better performance.
        num_selected: Number of individuals to select (int > 0)
        key: JAX PRNG key for randomness
        
    Returns:
        JAX array of selected individual indices
    """
    n_individuals, n_cases = fitness_matrix.shape
    
    def select_one_individual(i, state):
        """Select one individual using lexicase selection."""
        key, selected_so_far = state
        key, subkey = jax.random.split(key)
        
        # Shuffle the order of test cases
        case_order = jax.random.permutation(subkey, n_cases)
        
        # Start with all individuals as candidates
        candidates_mask = jnp.ones(n_individuals, dtype=bool)
        
        # Filter candidates case by case
        def filter_by_case(case_idx, candidates_mask):
            # Count remaining candidates
            num_candidates = jnp.sum(candidates_mask)
            
            # If only one or no candidates left, stop filtering
            def continue_filtering():
                case_fitness = jnp.where(
                    candidates_mask,
                    fitness_matrix[:, case_idx],
                    -jnp.inf  # Exclude non-candidates
                )
                max_fitness = jnp.max(case_fitness)
                
                # Keep only individuals with maximum fitness on this case
                best_mask = case_fitness == max_fitness
                return candidates_mask & best_mask
            
            def stop_filtering():
                return candidates_mask
            
            return lax.cond(
                num_candidates <= 1,
                stop_filtering,
                continue_filtering
            )
        
        # Apply filtering for each case in shuffled order
        def scan_filter(mask, case_idx):
            return filter_by_case(case_idx, mask), None
        
        final_candidates_mask, _ = lax.scan(scan_filter, candidates_mask, case_order)
        
        # Get indices of remaining candidates
        candidate_indices = jnp.where(final_candidates_mask)[0]
        num_candidates = len(candidate_indices)
        
        # Randomly select one from remaining candidates
        key, subkey = jax.random.split(key)
        chosen_idx = jax.random.choice(subkey, num_candidates)
        selected_individual = candidate_indices[chosen_idx]
        
        # Update selected array
        new_selected = selected_so_far.at[i].set(selected_individual)
        
        return key, new_selected
    
    # Initialize selected array
    selected = jnp.zeros(num_selected, dtype=jnp.int32)
    
    # Perform selection for all individuals
    def scan_select(state, i):
        return select_one_individual(i, state), None
    
    (key, final_selected), _ = lax.scan(scan_select, (key, selected), jnp.arange(num_selected))
    
    return final_selected


# Apply JIT compilation with static argument
jax_lexicase_selection_impl = jax.jit(jax_lexicase_selection_impl, static_argnums=1)


# Keep the old function for backward compatibility but make it a wrapper
def jax_lexicase_selection(fitness_matrix, num_selected, key):
    """
    JAX-based lexicase selection implementation (wrapper).
    
    Args:
        fitness_matrix: JAX array of shape (n_individuals, n_cases)
        num_selected: Number of individuals to select
        key: JAX PRNG key for randomness
        
    Returns:
        JAX array of selected individual indices
    """
    if num_selected == 0:
        return jnp.array([], dtype=jnp.int32)
    
    return jax_lexicase_selection_impl(fitness_matrix, num_selected, key)


@jax.jit  
def jax_epsilon_lexicase_selection(fitness_matrix, num_selected, epsilon, key):
    """
    JAX-based epsilon lexicase selection implementation.
    
    Args:
        fitness_matrix: JAX array of shape (n_individuals, n_cases)
        num_selected: Number of individuals to select
        epsilon: Tolerance value(s). Can be scalar or array of length n_cases
        key: JAX PRNG key
        
    Returns:
        JAX array of selected individual indices
    """
    # Handle zero selection case using JAX conditionals
    def zero_selection():
        return jnp.array([], dtype=jnp.int32)
    
    def normal_selection():
        n_individuals, n_cases = fitness_matrix.shape
        
        # Handle epsilon - ensure it's the right shape
        epsilon_values = jnp.broadcast_to(epsilon, (n_cases,))
        
        def select_one_individual(i, state):
            """Select one individual using epsilon lexicase selection."""
            key, selected_so_far = state
            key, subkey = jax.random.split(key)
            
            # Shuffle the order of test cases
            case_order = jax.random.permutation(subkey, n_cases)
            
            # Start with all individuals as candidates
            candidates_mask = jnp.ones(n_individuals, dtype=bool)
            
            # Filter candidates case by case
            def filter_by_case(case_idx, candidates_mask):
                # Count remaining candidates
                num_candidates = jnp.sum(candidates_mask)
                
                # If only one or no candidates left, stop filtering
                def continue_filtering():
                    case_fitness = jnp.where(
                        candidates_mask,
                        fitness_matrix[:, case_idx],
                        -jnp.inf  # Exclude non-candidates
                    )
                    max_fitness = jnp.max(case_fitness)
                    case_epsilon = epsilon_values[case_idx]
                    
                    # Keep individuals within epsilon of the best performance
                    best_mask = case_fitness >= (max_fitness - case_epsilon)
                    return candidates_mask & best_mask
                
                def stop_filtering():
                    return candidates_mask
                
                return lax.cond(
                    num_candidates <= 1,
                    stop_filtering,
                    continue_filtering
                )
            
            # Apply filtering for each case in shuffled order
            final_candidates_mask = lax.foldl(filter_by_case, candidates_mask, case_order)
            
            # Get indices of remaining candidates
            candidate_indices = jnp.where(final_candidates_mask, size=n_individuals, fill_value=-1)[0]
            valid_candidates = candidate_indices[candidate_indices >= 0]
            
            # Randomly select one from remaining candidates
            key, subkey = jax.random.split(key)
            chosen_idx = jax.random.choice(subkey, len(valid_candidates))
            selected_individual = valid_candidates[chosen_idx]
            
            # Update selected array
            new_selected = selected_so_far.at[i].set(selected_individual)
            
            return key, new_selected
        
        # Initialize selected array
        selected = jnp.zeros(num_selected, dtype=jnp.int32)
        
        # Perform selection for all individuals
        key, final_selected = lax.foldl(
            select_one_individual, 
            (key, selected), 
            jnp.arange(num_selected)
        )
        
        return final_selected
    
    return lax.cond(num_selected == 0, zero_selection, normal_selection)


def jax_compute_mad_epsilon(fitness_matrix):
    """
    Compute Median Absolute Deviation (MAD) for each test case using JAX.
    
    Args:
        fitness_matrix: JAX array of shape (n_individuals, n_cases)
        
    Returns:
        JAX array of MAD values for each test case
    """
    # Calculate median for each case (column)
    case_medians = jnp.median(fitness_matrix, axis=0)
    
    # Calculate absolute deviations from median for each case
    abs_deviations = jnp.abs(fitness_matrix - case_medians[None, :])
    
    # Calculate median of absolute deviations for each case
    mad_values = jnp.median(abs_deviations, axis=0)
    
    # Handle case where MAD is 0 (all values identical) by using a small default
    min_epsilon = 1e-10
    mad_values = jnp.maximum(mad_values, min_epsilon)
    
    return mad_values


@jax.jit
def jax_epsilon_lexicase_selection_with_mad(fitness_matrix, num_selected, key):
    """
    JAX epsilon lexicase selection using MAD-based adaptive epsilon.
    
    Args:
        fitness_matrix: JAX array of shape (n_individuals, n_cases)
        num_selected: Number of individuals to select
        key: JAX PRNG key
        
    Returns:
        JAX array of selected individual indices
    """
    # Compute MAD-based epsilon values
    epsilon_values = jax_compute_mad_epsilon(fitness_matrix)
    
    # Use epsilon lexicase with computed epsilon
    return jax_epsilon_lexicase_selection(fitness_matrix, num_selected, epsilon_values, key)