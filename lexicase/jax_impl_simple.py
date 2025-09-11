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


def _jax_compute_case_distances(fitness_matrix, sample_indices, threshold=None):
    """
    Compute pairwise distances between test cases based on solve patterns using JAX.
    
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
        thresholds = jnp.median(sampled_fitness, axis=0)
        solve_matrix = sampled_fitness > thresholds[None, :]
    elif jnp.isscalar(threshold):
        # Use single threshold for all
        solve_matrix = sampled_fitness > threshold
    else:
        # Use per-case thresholds
        solve_matrix = sampled_fitness > threshold[None, :]
    
    # Compute Hamming distances between cases using broadcasting
    # Expand solve matrix for pairwise comparison
    solve_expanded_i = solve_matrix[:, :, None]  # Shape: (n_samples, n_cases, 1)
    solve_expanded_j = solve_matrix[:, None, :]  # Shape: (n_samples, 1, n_cases)
    
    # Compute pairwise differences and sum across samples
    differences = solve_expanded_i != solve_expanded_j  # Shape: (n_samples, n_cases, n_cases)
    distances = jnp.sum(differences, axis=0)  # Shape: (n_cases, n_cases)
    
    return distances


def _jax_farthest_first_traversal(distances, downsample_size, key):
    """
    Select cases using Farthest First Traversal algorithm in JAX.
    
    Note: This is a simplified version that may not be fully JIT-compatible
    due to dynamic selection. For production, consider a fixed-iteration approach.
    
    Args:
        distances: Pairwise distance matrix between cases (n_cases, n_cases)
        downsample_size: Number of cases to select
        key: JAX random key
        
    Returns:
        Array of selected case indices
    """
    n_cases = distances.shape[0]
    
    # Handle edge cases
    if downsample_size >= n_cases:
        return jnp.arange(n_cases)
    
    # For JAX compatibility, we'll use a fixed-size approach
    selected_mask = jnp.zeros(n_cases, dtype=bool)
    selected_indices = jnp.full(downsample_size, -1, dtype=jnp.int32)
    
    # Randomly select first case
    key, subkey = jax.random.split(key)
    first_idx = jax.random.randint(subkey, (), 0, n_cases)
    selected_mask = selected_mask.at[first_idx].set(True)
    selected_indices = selected_indices.at[0].set(first_idx)
    
    # For remaining selections
    for i in range(1, downsample_size):
        # Compute minimum distances to selected cases
        # Use large negative value for already selected cases
        masked_distances = jnp.where(
            selected_mask[:, None],
            distances,
            jnp.full_like(distances, jnp.inf)
        )
        min_distances = jnp.min(masked_distances, axis=0)
        
        # Mask out already selected cases
        min_distances = jnp.where(selected_mask, -jnp.inf, min_distances)
        
        # Find maximum of minimum distances
        max_min_dist = jnp.max(min_distances)
        
        # Find all cases with this distance (handle ties)
        candidates_mask = (min_distances == max_min_dist)
        n_candidates = jnp.sum(candidates_mask)
        
        # Select randomly among ties
        key, subkey = jax.random.split(key)
        if n_candidates > 0:
            # Get candidate indices
            candidate_indices = jnp.where(candidates_mask, size=n_cases, fill_value=-1)[0]
            chosen_offset = jax.random.randint(subkey, (), 0, n_candidates)
            chosen_idx = candidate_indices[chosen_offset]
        else:
            # Fallback: randomly select from unselected
            unselected_mask = ~selected_mask
            unselected_indices = jnp.where(unselected_mask, size=n_cases, fill_value=-1)[0]
            n_unselected = jnp.sum(unselected_mask)
            chosen_offset = jax.random.randint(subkey, (), 0, n_unselected)
            chosen_idx = unselected_indices[chosen_offset]
        
        selected_mask = selected_mask.at[chosen_idx].set(True)
        selected_indices = selected_indices.at[i].set(chosen_idx)
    
    return selected_indices


def jax_informed_downsample_lexicase_selection_impl(
    fitness_matrix, num_selected, downsample_size, key,
    sample_rate=0.01, threshold=None, elitism=0
):
    """
    JAX-based informed downsampled lexicase selection implementation.
    
    Uses population statistics to select informative test cases rather than
    random sampling.
    
    Args:
        fitness_matrix: JAX array of shape (n_individuals, n_cases)
        num_selected: Number of individuals to select
        downsample_size: Number of test cases to select for each selection
        key: JAX random key
        sample_rate: Fraction of population to sample for distance calculation
        threshold: Optional threshold for pass/fail. If None, uses median.
        elitism: Number of best individuals to always include (by total fitness)
        
    Returns:
        JAX array of selected individual indices
    """
    n_individuals, n_cases = fitness_matrix.shape
    actual_downsample_size = jnp.minimum(downsample_size, n_cases)
    
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
    
    # Sample individuals for distance calculation
    key, subkey = jax.random.split(key)
    n_samples = jnp.maximum(1, jnp.int32(n_individuals * sample_rate))
    sample_indices = jax.random.choice(subkey, n_individuals, shape=(n_samples,), replace=False)
    
    # Compute case distances based on sampled individuals
    distances = _jax_compute_case_distances(fitness_matrix, sample_indices, threshold)
    
    # Select informative cases using Farthest First Traversal
    key, subkey = jax.random.split(key)
    informative_cases = _jax_farthest_first_traversal(distances, actual_downsample_size, subkey)
    
    # Create submatrix with only informative cases
    submatrix = fitness_matrix[:, informative_cases]
    
    # Perform selection for remaining slots using a loop
    def loop_cond(state):
        i, _, _ = state
        return i < num_selected
    
    def loop_body(state):
        i, selected_array, key = state
        
        # Shuffle case order for the submatrix
        key, subkey = jax.random.split(key)
        case_order = jax.random.permutation(subkey, actual_downsample_size)
        
        # Start with all individuals as candidates (using mask)
        candidates_mask = jnp.ones(n_individuals, dtype=bool)
        
        def filter_step(j, candidates_mask):
            case_idx = case_order[j]
            
            # Only proceed if we have more than one candidate
            n_candidates = jnp.sum(candidates_mask)
            
            def do_filtering():
                # Get fitness for this case among current candidates
                case_fitness = jnp.where(candidates_mask, submatrix[:, case_idx], -jnp.inf)
                max_fitness = jnp.max(case_fitness)
                
                # Create new mask for best performers
                best_mask = (case_fitness == max_fitness) & candidates_mask
                return best_mask
            
            def skip_filtering():
                return candidates_mask
            
            # Only filter if we have more than one candidate
            new_mask = lax.cond(n_candidates > 1, do_filtering, skip_filtering)
            
            return new_mask
        
        final_mask = lax.fori_loop(0, actual_downsample_size, filter_step, candidates_mask)
        
        # Select randomly from remaining candidates
        key, subkey = jax.random.split(key)
        candidates_indices = jnp.where(final_mask, size=n_individuals, fill_value=-1)[0]
        n_valid = jnp.sum(final_mask)
        
        # Choose random index from valid candidates
        chosen_idx = jax.random.randint(subkey, (), 0, n_valid)
        selected = candidates_indices[chosen_idx]
        
        selected_array = selected_array.at[i].set(selected)
        return i + 1, selected_array, key
    
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


def _jax_compute_case_distances(fitness_matrix, sample_indices, threshold=None):
    """
    Compute pairwise distances between test cases based on solve patterns using JAX.
    
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
        thresholds = jnp.median(sampled_fitness, axis=0)
        solve_matrix = sampled_fitness > thresholds[None, :]
    elif jnp.isscalar(threshold):
        # Use single threshold for all
        solve_matrix = sampled_fitness > threshold
    else:
        # Use per-case thresholds
        solve_matrix = sampled_fitness > threshold[None, :]
    
    # Compute Hamming distances between cases using broadcasting
    # Expand solve matrix for pairwise comparison
    solve_expanded_i = solve_matrix[:, :, None]  # Shape: (n_samples, n_cases, 1)
    solve_expanded_j = solve_matrix[:, None, :]  # Shape: (n_samples, 1, n_cases)
    
    # Compute pairwise differences and sum across samples
    differences = solve_expanded_i != solve_expanded_j  # Shape: (n_samples, n_cases, n_cases)
    distances = jnp.sum(differences, axis=0)  # Shape: (n_cases, n_cases)
    
    return distances


def _jax_farthest_first_traversal(distances, downsample_size, key):
    """
    Select cases using Farthest First Traversal algorithm in JAX.
    
    Note: This is a simplified version that may not be fully JIT-compatible
    due to dynamic selection. For production, consider a fixed-iteration approach.
    
    Args:
        distances: Pairwise distance matrix between cases (n_cases, n_cases)
        downsample_size: Number of cases to select
        key: JAX random key
        
    Returns:
        Array of selected case indices
    """
    n_cases = distances.shape[0]
    
    # Handle edge cases
    if downsample_size >= n_cases:
        return jnp.arange(n_cases)
    
    # For JAX compatibility, we'll use a fixed-size approach
    selected_mask = jnp.zeros(n_cases, dtype=bool)
    selected_indices = jnp.full(downsample_size, -1, dtype=jnp.int32)
    
    # Randomly select first case
    key, subkey = jax.random.split(key)
    first_idx = jax.random.randint(subkey, (), 0, n_cases)
    selected_mask = selected_mask.at[first_idx].set(True)
    selected_indices = selected_indices.at[0].set(first_idx)
    
    # For remaining selections
    for i in range(1, downsample_size):
        # Compute minimum distances to selected cases
        # Use large negative value for already selected cases
        masked_distances = jnp.where(
            selected_mask[:, None],
            distances,
            jnp.full_like(distances, jnp.inf)
        )
        min_distances = jnp.min(masked_distances, axis=0)
        
        # Mask out already selected cases
        min_distances = jnp.where(selected_mask, -jnp.inf, min_distances)
        
        # Find maximum of minimum distances
        max_min_dist = jnp.max(min_distances)
        
        # Find all cases with this distance (handle ties)
        candidates_mask = (min_distances == max_min_dist)
        n_candidates = jnp.sum(candidates_mask)
        
        # Select randomly among ties
        key, subkey = jax.random.split(key)
        if n_candidates > 0:
            # Get candidate indices
            candidate_indices = jnp.where(candidates_mask, size=n_cases, fill_value=-1)[0]
            chosen_offset = jax.random.randint(subkey, (), 0, n_candidates)
            chosen_idx = candidate_indices[chosen_offset]
        else:
            # Fallback: randomly select from unselected
            unselected_mask = ~selected_mask
            unselected_indices = jnp.where(unselected_mask, size=n_cases, fill_value=-1)[0]
            n_unselected = jnp.sum(unselected_mask)
            chosen_offset = jax.random.randint(subkey, (), 0, n_unselected)
            chosen_idx = unselected_indices[chosen_offset]
        
        selected_mask = selected_mask.at[chosen_idx].set(True)
        selected_indices = selected_indices.at[i].set(chosen_idx)
    
    return selected_indices


def jax_informed_downsample_lexicase_selection_impl(
    fitness_matrix, num_selected, downsample_size, key,
    sample_rate=0.01, threshold=None, elitism=0
):
    """
    JAX-based informed downsampled lexicase selection implementation.
    
    Uses population statistics to select informative test cases rather than
    random sampling.
    
    Args:
        fitness_matrix: JAX array of shape (n_individuals, n_cases)
        num_selected: Number of individuals to select
        downsample_size: Number of test cases to select for each selection
        key: JAX random key
        sample_rate: Fraction of population to sample for distance calculation
        threshold: Optional threshold for pass/fail. If None, uses median.
        elitism: Number of best individuals to always include (by total fitness)
        
    Returns:
        JAX array of selected individual indices
    """
    n_individuals, n_cases = fitness_matrix.shape
    actual_downsample_size = jnp.minimum(downsample_size, n_cases)
    
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
    
    # Sample individuals for distance calculation
    key, subkey = jax.random.split(key)
    n_samples = jnp.maximum(1, jnp.int32(n_individuals * sample_rate))
    sample_indices = jax.random.choice(subkey, n_individuals, shape=(n_samples,), replace=False)
    
    # Compute case distances based on sampled individuals
    distances = _jax_compute_case_distances(fitness_matrix, sample_indices, threshold)
    
    # Select informative cases using Farthest First Traversal
    key, subkey = jax.random.split(key)
    informative_cases = _jax_farthest_first_traversal(distances, actual_downsample_size, subkey)
    
    # Create submatrix with only informative cases
    submatrix = fitness_matrix[:, informative_cases]
    
    # Perform selection for remaining slots using a loop
    def loop_cond(state):
        i, _, _ = state
        return i < num_selected
    
    def loop_body(state):
        i, selected_array, key = state
        
        # Shuffle case order for the submatrix
        key, subkey = jax.random.split(key)
        case_order = jax.random.permutation(subkey, actual_downsample_size)
        
        # Start with all individuals as candidates (using mask)
        candidates_mask = jnp.ones(n_individuals, dtype=bool)
        
        def filter_step(j, candidates_mask):
            case_idx = case_order[j]
            
            # Only proceed if we have more than one candidate
            n_candidates = jnp.sum(candidates_mask)
            
            def do_filtering():
                # Get fitness for this case among current candidates
                case_fitness = jnp.where(candidates_mask, submatrix[:, case_idx], -jnp.inf)
                max_fitness = jnp.max(case_fitness)
                
                # Create new mask for best performers
                best_mask = (case_fitness == max_fitness) & candidates_mask
                return best_mask
            
            def skip_filtering():
                return candidates_mask
            
            # Only filter if we have more than one candidate
            new_mask = lax.cond(n_candidates > 1, do_filtering, skip_filtering)
            
            return new_mask
        
        final_mask = lax.fori_loop(0, actual_downsample_size, filter_step, candidates_mask)
        
        # Select randomly from remaining candidates
        key, subkey = jax.random.split(key)
        candidates_indices = jnp.where(final_mask, size=n_individuals, fill_value=-1)[0]
        n_valid = jnp.sum(final_mask)
        
        # Choose random index from valid candidates
        chosen_idx = jax.random.randint(subkey, (), 0, n_valid)
        selected = candidates_indices[chosen_idx]
        
        selected_array = selected_array.at[i].set(selected)
        return i + 1, selected_array, key
    
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


def _jax_compute_case_distances(fitness_matrix, sample_indices, threshold=None):
    """
    Compute pairwise distances between test cases based on solve patterns using JAX.
    
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
        thresholds = jnp.median(sampled_fitness, axis=0)
        solve_matrix = sampled_fitness > thresholds[None, :]
    elif jnp.isscalar(threshold):
        # Use single threshold for all
        solve_matrix = sampled_fitness > threshold
    else:
        # Use per-case thresholds
        solve_matrix = sampled_fitness > threshold[None, :]
    
    # Compute Hamming distances between cases using broadcasting
    # Expand solve matrix for pairwise comparison
    solve_expanded_i = solve_matrix[:, :, None]  # Shape: (n_samples, n_cases, 1)
    solve_expanded_j = solve_matrix[:, None, :]  # Shape: (n_samples, 1, n_cases)
    
    # Compute pairwise differences and sum across samples
    differences = solve_expanded_i != solve_expanded_j  # Shape: (n_samples, n_cases, n_cases)
    distances = jnp.sum(differences, axis=0)  # Shape: (n_cases, n_cases)
    
    return distances


def _jax_farthest_first_traversal(distances, downsample_size, key):
    """
    Select cases using Farthest First Traversal algorithm in JAX.
    
    Note: This is a simplified version that may not be fully JIT-compatible
    due to dynamic selection. For production, consider a fixed-iteration approach.
    
    Args:
        distances: Pairwise distance matrix between cases (n_cases, n_cases)
        downsample_size: Number of cases to select
        key: JAX random key
        
    Returns:
        Array of selected case indices
    """
    n_cases = distances.shape[0]
    
    # Handle edge cases
    if downsample_size >= n_cases:
        return jnp.arange(n_cases)
    
    # For JAX compatibility, we'll use a fixed-size approach
    selected_mask = jnp.zeros(n_cases, dtype=bool)
    selected_indices = jnp.full(downsample_size, -1, dtype=jnp.int32)
    
    # Randomly select first case
    key, subkey = jax.random.split(key)
    first_idx = jax.random.randint(subkey, (), 0, n_cases)
    selected_mask = selected_mask.at[first_idx].set(True)
    selected_indices = selected_indices.at[0].set(first_idx)
    
    # For remaining selections
    for i in range(1, downsample_size):
        # Compute minimum distances to selected cases
        # Use large negative value for already selected cases
        masked_distances = jnp.where(
            selected_mask[:, None],
            distances,
            jnp.full_like(distances, jnp.inf)
        )
        min_distances = jnp.min(masked_distances, axis=0)
        
        # Mask out already selected cases
        min_distances = jnp.where(selected_mask, -jnp.inf, min_distances)
        
        # Find maximum of minimum distances
        max_min_dist = jnp.max(min_distances)
        
        # Find all cases with this distance (handle ties)
        candidates_mask = (min_distances == max_min_dist)
        n_candidates = jnp.sum(candidates_mask)
        
        # Select randomly among ties
        key, subkey = jax.random.split(key)
        if n_candidates > 0:
            # Get candidate indices
            candidate_indices = jnp.where(candidates_mask, size=n_cases, fill_value=-1)[0]
            chosen_offset = jax.random.randint(subkey, (), 0, n_candidates)
            chosen_idx = candidate_indices[chosen_offset]
        else:
            # Fallback: randomly select from unselected
            unselected_mask = ~selected_mask
            unselected_indices = jnp.where(unselected_mask, size=n_cases, fill_value=-1)[0]
            n_unselected = jnp.sum(unselected_mask)
            chosen_offset = jax.random.randint(subkey, (), 0, n_unselected)
            chosen_idx = unselected_indices[chosen_offset]
        
        selected_mask = selected_mask.at[chosen_idx].set(True)
        selected_indices = selected_indices.at[i].set(chosen_idx)
    
    return selected_indices


def jax_informed_downsample_lexicase_selection_impl(
    fitness_matrix, num_selected, downsample_size, key,
    sample_rate=0.01, threshold=None, elitism=0
):
    """
    JAX-based informed downsampled lexicase selection implementation.
    
    Uses population statistics to select informative test cases rather than
    random sampling.
    
    Args:
        fitness_matrix: JAX array of shape (n_individuals, n_cases)
        num_selected: Number of individuals to select
        downsample_size: Number of test cases to select for each selection
        key: JAX random key
        sample_rate: Fraction of population to sample for distance calculation
        threshold: Optional threshold for pass/fail. If None, uses median.
        elitism: Number of best individuals to always include (by total fitness)
        
    Returns:
        JAX array of selected individual indices
    """
    n_individuals, n_cases = fitness_matrix.shape
    actual_downsample_size = jnp.minimum(downsample_size, n_cases)
    
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
    
    # Sample individuals for distance calculation
    key, subkey = jax.random.split(key)
    n_samples = jnp.maximum(1, jnp.int32(n_individuals * sample_rate))
    sample_indices = jax.random.choice(subkey, n_individuals, shape=(n_samples,), replace=False)
    
    # Compute case distances based on sampled individuals
    distances = _jax_compute_case_distances(fitness_matrix, sample_indices, threshold)
    
    # Select informative cases using Farthest First Traversal
    key, subkey = jax.random.split(key)
    informative_cases = _jax_farthest_first_traversal(distances, actual_downsample_size, subkey)
    
    # Create submatrix with only informative cases
    submatrix = fitness_matrix[:, informative_cases]
    
    # Perform selection for remaining slots using a loop
    def loop_cond(state):
        i, _, _ = state
        return i < num_selected
    
    def loop_body(state):
        i, selected_array, key = state
        
        # Shuffle case order for the submatrix
        key, subkey = jax.random.split(key)
        case_order = jax.random.permutation(subkey, actual_downsample_size)
        
        # Start with all individuals as candidates (using mask)
        candidates_mask = jnp.ones(n_individuals, dtype=bool)
        
        def filter_step(j, candidates_mask):
            case_idx = case_order[j]
            
            # Only proceed if we have more than one candidate
            n_candidates = jnp.sum(candidates_mask)
            
            def do_filtering():
                # Get fitness for this case among current candidates
                case_fitness = jnp.where(candidates_mask, submatrix[:, case_idx], -jnp.inf)
                max_fitness = jnp.max(case_fitness)
                
                # Create new mask for best performers
                best_mask = (case_fitness == max_fitness) & candidates_mask
                return best_mask
            
            def skip_filtering():
                return candidates_mask
            
            # Only filter if we have more than one candidate
            new_mask = lax.cond(n_candidates > 1, do_filtering, skip_filtering)
            
            return new_mask
        
        final_mask = lax.fori_loop(0, actual_downsample_size, filter_step, candidates_mask)
        
        # Select randomly from remaining candidates
        key, subkey = jax.random.split(key)
        candidates_indices = jnp.where(final_mask, size=n_individuals, fill_value=-1)[0]
        n_valid = jnp.sum(final_mask)
        
        # Choose random index from valid candidates
        chosen_idx = jax.random.randint(subkey, (), 0, n_valid)
        selected = candidates_indices[chosen_idx]
        
        selected_array = selected_array.at[i].set(selected)
        return i + 1, selected_array, key
    
    final_i, final_selected, final_key = lax.while_loop(
        loop_cond, loop_body, (start_idx, selected_array, key)
    )
    
    return final_selected
