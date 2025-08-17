"""
Automatic dispatch layer for lexicase selection algorithms.

This module provides array-type-aware dispatch that automatically routes
to JAX or NumPy implementations based on the input array type.
"""

import numpy as np

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def _is_jax_array(arr):
    """
    Check if an array is a JAX array.
    
    Args:
        arr: Array to check
        
    Returns:
        bool: True if it's a JAX array, False otherwise
    """
    if not JAX_AVAILABLE:
        return False
    
    # Check if it's a JAX array by looking at the type
    return hasattr(arr, 'device') and hasattr(arr, 'at')


def _validate_and_convert_fitness_matrix(fitness_matrix):
    """
    Validate fitness matrix and ensure it's a proper array.
    
    Args:
        fitness_matrix: Input fitness matrix
        
    Returns:
        tuple: (validated_array, is_jax_array)
        
    Raises:
        ValueError: If matrix format is invalid
    """
    # Convert to appropriate array type
    if _is_jax_array(fitness_matrix):
        if JAX_AVAILABLE:
            fitness_matrix = jnp.asarray(fitness_matrix)
            is_jax = True
        else:
            raise ImportError("JAX array provided but JAX is not available")
    else:
        fitness_matrix = np.asarray(fitness_matrix)
        is_jax = False
    
    # Validate dimensions
    if fitness_matrix.ndim != 2:
        raise ValueError("Fitness matrix must be 2-dimensional")
    
    if fitness_matrix.shape[0] == 0:
        raise ValueError("Fitness matrix must have at least one individual")
    
    if fitness_matrix.shape[1] == 0:
        raise ValueError("Fitness matrix must have at least one test case")
    
    return fitness_matrix, is_jax


def _validate_selection_params(num_selected, seed=None):
    """
    Validate selection parameters.
    
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


def lexicase_selection(fitness_matrix, num_selected, seed=None):
    """
    Lexicase selection with automatic dispatch based on array type.
    
    Automatically detects whether the input is a JAX or NumPy array and
    routes to the appropriate high-performance implementation.
    
    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases) containing
                       fitness values. Higher values indicate better performance.
                       Can be NumPy array or JAX array.
        num_selected: Number of individuals to select
        seed: Random seed for reproducibility
        
    Returns:
        Array of selected individual indices (same type as input)
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    fitness_matrix, is_jax = _validate_and_convert_fitness_matrix(fitness_matrix)
    _validate_selection_params(num_selected, seed)
    
    if is_jax and JAX_AVAILABLE:
        # Use JAX implementation
        # Handle zero case outside JIT for better performance
        if num_selected == 0:
            return jnp.array([], dtype=jnp.int32)
        
        from .jax_impl_simple import jax_lexicase_selection_impl
        key = jax.random.PRNGKey(seed or 0)
        return jax_lexicase_selection_impl(fitness_matrix, num_selected, key)
    else:
        # Use NumPy implementation  
        from .numpy_impl import numpy_lexicase_selection
        rng = np.random.default_rng(seed)
        return numpy_lexicase_selection(fitness_matrix, num_selected, rng)


def epsilon_lexicase_selection(fitness_matrix, num_selected, epsilon=None, seed=None):
    """
    Epsilon lexicase selection with automatic dispatch based on array type.
    
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
        Array of selected individual indices (same type as input)
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    fitness_matrix, is_jax = _validate_and_convert_fitness_matrix(fitness_matrix)
    _validate_selection_params(num_selected, seed)
    
    if is_jax and JAX_AVAILABLE:
        # Use JAX implementation
        if epsilon is None:
            # Use MAD-based epsilon
            from .jax_impl_simple import jax_epsilon_lexicase_selection_with_mad
            key = jax.random.PRNGKey(seed or 0)
            return jax_epsilon_lexicase_selection_with_mad(fitness_matrix, num_selected, key)
        else:
            # Use provided epsilon
            from .jax_impl_simple import jax_epsilon_lexicase_selection_impl
            epsilon_jax = jnp.asarray(epsilon)
            
            # Validate epsilon (same as NumPy)
            if epsilon_jax.ndim > 0 and len(epsilon_jax) != fitness_matrix.shape[1]:
                raise ValueError(f"Epsilon array length ({len(epsilon_jax)}) must match number of cases ({fitness_matrix.shape[1]})")
            if jnp.any(epsilon_jax < 0):
                if epsilon_jax.ndim == 0:  # Scalar
                    raise ValueError("Epsilon must be non-negative")
                else:  # Array
                    raise ValueError("All epsilon values must be non-negative")
            
            key = jax.random.PRNGKey(seed or 0)
            return jax_epsilon_lexicase_selection_impl(fitness_matrix, num_selected, epsilon_jax, key)
    else:
        # Use NumPy implementation
        if epsilon is None:
            # Use MAD-based epsilon
            from .numpy_impl import numpy_epsilon_lexicase_selection_with_mad
            rng = np.random.default_rng(seed)
            return numpy_epsilon_lexicase_selection_with_mad(fitness_matrix, num_selected, rng)
        else:
            # Use provided epsilon
            from .numpy_impl import numpy_epsilon_lexicase_selection
            epsilon_np = np.asarray(epsilon)
            
            # Validate epsilon
            if epsilon_np.ndim > 0 and len(epsilon_np) != fitness_matrix.shape[1]:
                raise ValueError(f"Epsilon array length ({len(epsilon_np)}) must match number of cases ({fitness_matrix.shape[1]})")
            if np.any(epsilon_np < 0):
                if epsilon_np.ndim == 0:  # Scalar
                    raise ValueError("Epsilon must be non-negative")
                else:  # Array
                    raise ValueError("All epsilon values must be non-negative")
            
            rng = np.random.default_rng(seed)
            return numpy_epsilon_lexicase_selection(fitness_matrix, num_selected, epsilon_np, rng)


def downsample_lexicase_selection(fitness_matrix, num_selected, downsample_size, seed=None):
    """
    Downsampled lexicase selection with automatic dispatch based on array type.
    
    Args:
        fitness_matrix: Array of shape (n_individuals, n_cases) containing
                       fitness values. Higher values indicate better performance.
        num_selected: Number of individuals to select
        downsample_size: Number of test cases to randomly sample for each selection
        seed: Random seed for reproducibility
        
    Returns:
        Array of selected individual indices (same type as input)
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    fitness_matrix, is_jax = _validate_and_convert_fitness_matrix(fitness_matrix)
    _validate_selection_params(num_selected, seed)
    
    if downsample_size <= 0:
        raise ValueError("Downsample size must be positive")
    
    if is_jax and JAX_AVAILABLE:
        # Use JAX implementation
        from .jax_impl_simple import jax_downsample_lexicase_selection_impl
        key = jax.random.PRNGKey(seed or 0)
        return jax_downsample_lexicase_selection_impl(fitness_matrix, num_selected, downsample_size, key)
    else:
        # Use NumPy implementation
        from .numpy_impl import numpy_downsample_lexicase_selection
        rng = np.random.default_rng(seed)
        return numpy_downsample_lexicase_selection(fitness_matrix, num_selected, downsample_size, rng)


