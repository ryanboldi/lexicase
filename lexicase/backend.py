"""
Backend management for lexicase selection.

Provides abstraction layer for switching between NumPy and JAX backends.
"""

_BACKEND = "numpy"

def set_backend(backend: str):
    """Set the backend for lexicase operations.
    
    Args:
        backend: Either 'numpy' or 'jax'
        
    Raises:
        ValueError: If backend is not supported
    """
    global _BACKEND
    if backend not in ["numpy", "jax"]:
        raise ValueError("Backend must be 'numpy' or 'jax'")
    _BACKEND = backend

def get_backend():
    """Get the current backend.
    
    Returns:
        str: Current backend name
    """
    return _BACKEND

def get_lib(seed=None):
    """Get the appropriate array library and random number generator.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (array_library, random_number_generator)
    """
    if _BACKEND == "jax":
        try:
            import jax
            import jax.numpy as xp
            rng = jax.random.PRNGKey(seed) if seed is not None else None
        except ImportError:
            raise ImportError("JAX backend selected but JAX is not installed. Install with: pip install .[jax]")
    else:
        try:
            import numpy as xp
            rng = xp.random.default_rng(seed) if seed is not None else None
        except ImportError:
            raise ImportError("NumPy backend selected but NumPy is not installed. Install with: pip install .[numpy]")
    return xp, rng 