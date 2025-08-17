"""
Simplified JAX implementation that avoids complex JIT issues.

For now, let's create a simpler version that works and optimize later.
"""

import jax
import jax.numpy as jnp
from jax import lax


def _lexicase_single_selection(fitness_matrix, key):
    """Select a single individual using lexicase selection (JIT-friendly)."""
    n_individuals, n_cases = fitness_matrix.shape
    
    # Shuffle case order
    key, subkey = jax.random.split(key)
    case_order = jax.random.permutation(subkey, n_cases)
    
    # Start with all individuals as candidates (using mask)
    candidates_mask = jnp.ones(n_individuals, dtype=bool)
    
    def filter_step(i, state):
        candidates_mask, key = state
        case_idx = case_order[i]
        
        # Only proceed if we have more than one candidate
        n_candidates = jnp.sum(candidates_mask)
        
        def do_filtering():
            # Get fitness for this case among current candidates
            case_fitness = jnp.where(candidates_mask, fitness_matrix[:, case_idx], -jnp.inf)
            max_fitness = jnp.max(case_fitness)
            
            # Create new mask for best performers
            best_mask = (case_fitness == max_fitness) & candidates_mask
            return best_mask
        
        def skip_filtering():
            return candidates_mask
        
        # Only filter if we have more than one candidate
        new_mask = lax.cond(n_candidates > 1, do_filtering, skip_filtering)
        
        return new_mask, key
    
    final_mask, _ = lax.fori_loop(0, n_cases, filter_step, (candidates_mask, key))
    
    # Select randomly from remaining candidates
    key, subkey = jax.random.split(key)
    candidates_indices = jnp.where(final_mask, size=n_individuals, fill_value=-1)[0]
    n_valid = jnp.sum(final_mask)
    
    # Choose random index from valid candidates
    chosen_idx = jax.random.randint(subkey, (), 0, n_valid)
    selected = candidates_indices[chosen_idx]
    
    return selected, key


def jax_lexicase_selection_impl(fitness_matrix, num_selected, key, elitism=0):
    """
    JIT-friendly JAX lexicase selection implementation.
    
    Uses a while loop to handle dynamic num_selected.
    """
    n_individuals, n_cases = fitness_matrix.shape
    
    # Create output array with exact size needed
    selected_array = jnp.full(num_selected, -1, dtype=jnp.int32)
    
    # Handle elitism: select best individuals by total fitness
    start_idx = 0
    if elitism > 0:
        # Calculate total fitness for each individual
        total_fitness = jnp.sum(fitness_matrix, axis=1)
        # Get indices of top performers
        elite_indices = jnp.argsort(total_fitness)[-elitism:]
        # Add elite individuals to selection
        selected_array = selected_array.at[:elitism].set(elite_indices)
        start_idx = elitism
    
    def loop_cond(state):
        i, _, _ = state
        return i < num_selected
    
    def loop_body(state):
        i, selected_array, key = state
        selected, new_key = _lexicase_single_selection(fitness_matrix, key)
        selected_array = selected_array.at[i].set(selected)
        return i + 1, selected_array, new_key
    
    final_i, final_selected, final_key = lax.while_loop(
        loop_cond, loop_body, (start_idx, selected_array, key)
    )
    
    return final_selected


def _epsilon_lexicase_single_selection(fitness_matrix, epsilon_values, key):
    """Select a single individual using epsilon lexicase selection (JIT-friendly)."""
    n_individuals, n_cases = fitness_matrix.shape
    
    # Shuffle case order
    key, subkey = jax.random.split(key)
    case_order = jax.random.permutation(subkey, n_cases)
    
    # Start with all individuals as candidates (using mask)
    candidates_mask = jnp.ones(n_individuals, dtype=bool)
    
    def filter_step(i, state):
        candidates_mask, key = state
        case_idx = case_order[i]
        
        # Only proceed if we have more than one candidate
        n_candidates = jnp.sum(candidates_mask)
        
        def do_filtering():
            # Get fitness for this case among current candidates
            case_fitness = jnp.where(candidates_mask, fitness_matrix[:, case_idx], -jnp.inf)
            max_fitness = jnp.max(case_fitness)
            case_epsilon = epsilon_values[case_idx]
            
            # Create new mask for individuals within epsilon of best
            best_mask = (case_fitness >= (max_fitness - case_epsilon)) & candidates_mask
            return best_mask
        
        def skip_filtering():
            return candidates_mask
        
        # Only filter if we have more than one candidate
        new_mask = lax.cond(n_candidates > 1, do_filtering, skip_filtering)
        
        return new_mask, key
    
    final_mask, _ = lax.fori_loop(0, n_cases, filter_step, (candidates_mask, key))
    
    # Select randomly from remaining candidates
    key, subkey = jax.random.split(key)
    candidates_indices = jnp.where(final_mask, size=n_individuals, fill_value=-1)[0]
    n_valid = jnp.sum(final_mask)
    
    # Choose random index from valid candidates
    chosen_idx = jax.random.randint(subkey, (), 0, n_valid)
    selected = candidates_indices[chosen_idx]
    
    return selected, key


def jax_epsilon_lexicase_selection_impl(fitness_matrix, num_selected, epsilon, key, elitism=0):
    """
    JIT-friendly JAX epsilon lexicase selection implementation.
    """
    n_individuals, n_cases = fitness_matrix.shape
    
    # Handle epsilon - ensure it's the right shape
    epsilon_values = jnp.broadcast_to(epsilon, (n_cases,))
    
    # Create output array with exact size needed
    selected_array = jnp.full(num_selected, -1, dtype=jnp.int32)
    
    # Handle elitism: select best individuals by total fitness
    start_idx = 0
    if elitism > 0:
        # Calculate total fitness for each individual
        total_fitness = jnp.sum(fitness_matrix, axis=1)
        # Get indices of top performers
        elite_indices = jnp.argsort(total_fitness)[-elitism:]
        # Add elite individuals to selection
        selected_array = selected_array.at[:elitism].set(elite_indices)
        start_idx = elitism
    
    def loop_cond(state):
        i, _, _ = state
        return i < num_selected
    
    def loop_body(state):
        i, selected_array, key = state
        selected, new_key = _epsilon_lexicase_single_selection(fitness_matrix, epsilon_values, key)
        selected_array = selected_array.at[i].set(selected)
        return i + 1, selected_array, new_key
    
    final_i, final_selected, final_key = lax.while_loop(
        loop_cond, loop_body, (start_idx, selected_array, key)
    )
    
    return final_selected


def jax_compute_mad_epsilon(fitness_matrix):
    """Compute MAD epsilon for JAX arrays."""
    case_medians = jnp.median(fitness_matrix, axis=0)
    abs_deviations = jnp.abs(fitness_matrix - case_medians[None, :])
    mad_values = jnp.median(abs_deviations, axis=0)
    min_epsilon = 1e-10
    mad_values = jnp.maximum(mad_values, min_epsilon)
    return mad_values


def jax_epsilon_lexicase_selection_with_mad(fitness_matrix, num_selected, key, elitism=0):
    """JAX epsilon lexicase with MAD-based epsilon."""
    epsilon_values = jax_compute_mad_epsilon(fitness_matrix)
    return jax_epsilon_lexicase_selection_impl(fitness_matrix, num_selected, epsilon_values, key, elitism)


def _downsample_lexicase_single_selection(fitness_matrix, downsample_size, key):
    """Select a single individual using downsampled lexicase selection (JIT-friendly)."""
    n_individuals, n_cases = fitness_matrix.shape
    
    # For JIT compatibility, we need downsample_size to be static
    # So we'll just use it directly and let the test handle edge cases
    
    # Randomly sample test cases for this selection
    key, subkey = jax.random.split(key)
    # Use permutation and take first downsample_size elements
    all_indices = jax.random.permutation(subkey, n_cases)
    
    # If downsample_size >= n_cases, we effectively use all cases
    # This is handled by taking minimum during case loop
    sampled_cases = all_indices[:downsample_size]
    
    # Create submatrix with only sampled cases
    submatrix = fitness_matrix[:, sampled_cases]
    
    # Shuffle case order for the submatrix
    key, subkey = jax.random.split(key)
    case_order = jax.random.permutation(subkey, downsample_size)
    
    # Start with all individuals as candidates (using mask)
    candidates_mask = jnp.ones(n_individuals, dtype=bool)
    
    def filter_step(i, state):
        candidates_mask, key = state
        
        # Only proceed if this case index is valid (< n_cases)
        # and we have more than one candidate
        n_candidates = jnp.sum(candidates_mask)
        case_idx = case_order[i]
        
        def do_filtering():
            # Only filter if case index is valid
            def valid_case_filter():
                case_fitness = jnp.where(candidates_mask, submatrix[:, case_idx], -jnp.inf)
                max_fitness = jnp.max(case_fitness)
                best_mask = (case_fitness == max_fitness) & candidates_mask
                return best_mask
            
            def invalid_case_skip():
                return candidates_mask
            
            # Check if case index is valid
            return lax.cond(case_idx < n_cases, valid_case_filter, invalid_case_skip)
        
        def skip_filtering():
            return candidates_mask
        
        # Only filter if we have more than one candidate
        new_mask = lax.cond(n_candidates > 1, do_filtering, skip_filtering)
        
        return new_mask, key
    
    final_mask, _ = lax.fori_loop(0, downsample_size, filter_step, (candidates_mask, key))
    
    # Select randomly from remaining candidates
    key, subkey = jax.random.split(key)
    candidates_indices = jnp.where(final_mask, size=n_individuals, fill_value=-1)[0]
    n_valid = jnp.sum(final_mask)
    
    # Choose random index from valid candidates
    chosen_idx = jax.random.randint(subkey, (), 0, n_valid)
    selected = candidates_indices[chosen_idx]
    
    return selected, key


def jax_downsample_lexicase_selection_impl(fitness_matrix, num_selected, downsample_size, key, elitism=0):
    """
    JIT-friendly JAX-based downsampled lexicase selection implementation.
    """
    n_individuals, n_cases = fitness_matrix.shape
    
    # Create output array with exact size needed
    selected_array = jnp.full(num_selected, -1, dtype=jnp.int32)
    
    # Handle elitism: select best individuals by total fitness
    start_idx = 0
    if elitism > 0:
        # Calculate total fitness for each individual
        total_fitness = jnp.sum(fitness_matrix, axis=1)
        # Get indices of top performers
        elite_indices = jnp.argsort(total_fitness)[-elitism:]
        # Add elite individuals to selection
        selected_array = selected_array.at[:elitism].set(elite_indices)
        start_idx = elitism
    
    def loop_cond(state):
        i, _, _ = state
        return i < num_selected
    
    def loop_body(state):
        i, selected_array, key = state
        selected, new_key = _downsample_lexicase_single_selection(fitness_matrix, downsample_size, key)
        selected_array = selected_array.at[i].set(selected)
        return i + 1, selected_array, new_key
    
    final_i, final_selected, final_key = lax.while_loop(
        loop_cond, loop_body, (start_idx, selected_array, key)
    )
    
    return final_selected
