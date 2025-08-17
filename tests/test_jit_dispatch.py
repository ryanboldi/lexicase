"""
Tests for JIT-friendly dispatch system.

Tests the new architecture where:
1. Pure JAX functions are JIT-compiled
2. Pure NumPy functions avoid JAX overhead  
3. Dispatch happens at the API level based on array type
4. No runtime backend switching or global state
"""

import numpy as np
import pytest

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Import the new API (will implement these)
try:
    from lexicase.jax_impl_simple import jax_lexicase_selection_impl as jax_lexicase_selection
    from lexicase.jax_impl_simple import jax_epsilon_lexicase_selection_impl as jax_epsilon_lexicase_selection
    from lexicase.numpy_impl import numpy_lexicase_selection, numpy_epsilon_lexicase_selection
    from lexicase import lexicase_selection, epsilon_lexicase_selection
except ImportError:
    # These don't exist yet - we'll implement them
    pass


class TestJITFriendlyDispatch:
    """Test the new JIT-friendly dispatch system."""
    
    def setup_method(self):
        """Set up test data."""
        self.fitness_matrix_np = np.array([
            [1.0, 0.0, 1.0],  # Individual 0: good at cases 0,2
            [0.0, 1.0, 0.0],  # Individual 1: good at case 1  
            [0.5, 0.5, 0.5],  # Individual 2: mediocre at all
        ])
        
        if JAX_AVAILABLE:
            self.fitness_matrix_jax = jnp.array(self.fitness_matrix_np)
    
    def test_numpy_array_dispatch(self):
        """Test that NumPy arrays are dispatched to NumPy implementation."""
        selected = lexicase_selection(self.fitness_matrix_np, num_selected=2, seed=42)
        
        # Should return NumPy array
        assert isinstance(selected, np.ndarray)
        assert len(selected) == 2
        assert all(0 <= idx <= 2 for idx in selected)
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_array_dispatch(self):
        """Test that JAX arrays are dispatched to JAX implementation."""
        selected = lexicase_selection(self.fitness_matrix_jax, num_selected=2, seed=42)
        
        # Should return JAX array
        assert hasattr(selected, 'device')  # JAX array characteristic
        assert len(selected) == 2
        assert all(0 <= idx <= 2 for idx in selected)
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_jit_compilation(self):
        """Test that JAX implementation can be JIT compiled."""
        # num_selected needs to be static for JIT compilation
        jitted_fn = jax.jit(jax_lexicase_selection, static_argnums=(1,))
        
        key = jax.random.PRNGKey(42)
        selected = jitted_fn(self.fitness_matrix_jax, 2, key)
        
        assert len(selected) == 2
        assert all(0 <= idx <= 2 for idx in selected)
    
    def test_deterministic_with_seed_numpy(self):
        """Test NumPy implementation is deterministic with seed."""
        selected1 = lexicase_selection(self.fitness_matrix_np, num_selected=5, seed=42)
        selected2 = lexicase_selection(self.fitness_matrix_np, num_selected=5, seed=42)
        
        np.testing.assert_array_equal(selected1, selected2)
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_deterministic_with_seed_jax(self):
        """Test JAX implementation is deterministic with seed."""
        selected1 = lexicase_selection(self.fitness_matrix_jax, num_selected=5, seed=42)
        selected2 = lexicase_selection(self.fitness_matrix_jax, num_selected=5, seed=42)
        
        np.testing.assert_array_equal(selected1, selected2)
    
    def test_epsilon_lexicase_numpy_dispatch(self):
        """Test epsilon lexicase with NumPy arrays."""
        selected = epsilon_lexicase_selection(
            self.fitness_matrix_np, 
            num_selected=2, 
            epsilon=0.1, 
            seed=42
        )
        
        assert isinstance(selected, np.ndarray)
        assert len(selected) == 2
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_epsilon_lexicase_jax_dispatch(self):
        """Test epsilon lexicase with JAX arrays."""
        selected = epsilon_lexicase_selection(
            self.fitness_matrix_jax, 
            num_selected=2, 
            epsilon=0.1, 
            seed=42
        )
        
        assert hasattr(selected, 'device')  # JAX array
        assert len(selected) == 2
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_epsilon_lexicase_jit_compilation(self):
        """Test that epsilon lexicase JAX implementation can be JIT compiled."""
        # num_selected needs to be static for JIT compilation
        jitted_fn = jax.jit(jax_epsilon_lexicase_selection, static_argnums=(1,))
        
        key = jax.random.PRNGKey(42)
        epsilon = jnp.array(0.1)
        selected = jitted_fn(self.fitness_matrix_jax, 2, epsilon, key)
        
        assert len(selected) == 2
    
    def test_edge_case_zero_selected(self):
        """Test edge case of selecting zero individuals."""
        selected_np = lexicase_selection(self.fitness_matrix_np, num_selected=0, seed=42)
        assert len(selected_np) == 0
        assert isinstance(selected_np, np.ndarray)
        
        if JAX_AVAILABLE:
            selected_jax = lexicase_selection(self.fitness_matrix_jax, num_selected=0, seed=42)
            assert len(selected_jax) == 0
            assert hasattr(selected_jax, 'device')
    
    def test_edge_case_single_individual(self):
        """Test edge case with single individual."""
        single_fitness_np = np.array([[1.0, 0.5, 0.8]])
        selected = lexicase_selection(single_fitness_np, num_selected=3, seed=42)
        
        # Should select the same individual multiple times
        assert len(selected) == 3
        assert all(idx == 0 for idx in selected)
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_performance_no_conversion_overhead(self):
        """Test that JAX arrays don't get converted unnecessarily."""
        # This is more of a design test - JAX arrays should stay JAX arrays
        result = lexicase_selection(self.fitness_matrix_jax, num_selected=1, seed=42)
        
        # Result should be a JAX array, not converted to NumPy
        assert hasattr(result, 'device')
        assert not isinstance(result, np.ndarray)


class TestPureImplementations:
    """Test the pure JAX and NumPy implementations directly."""
    
    def setup_method(self):
        """Set up test data."""
        self.fitness_matrix_np = np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0], 
            [0.5, 0.5, 0.5],
        ])
        
        if JAX_AVAILABLE:
            self.fitness_matrix_jax = jnp.array(self.fitness_matrix_np)
    
    def test_numpy_implementation_direct(self):
        """Test calling NumPy implementation directly."""
        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(self.fitness_matrix_np, 2, rng)
        
        assert isinstance(selected, np.ndarray)
        assert len(selected) == 2
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_implementation_direct(self):
        """Test calling JAX implementation directly."""
        key = jax.random.PRNGKey(42)
        selected = jax_lexicase_selection(self.fitness_matrix_jax, 2, key)
        
        assert hasattr(selected, 'device')
        assert len(selected) == 2
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_implementation_multiple_calls_different_keys(self):
        """Test that different JAX keys produce different results."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(43)
        
        selected1 = jax_lexicase_selection(self.fitness_matrix_jax, 10, key1)
        selected2 = jax_lexicase_selection(self.fitness_matrix_jax, 10, key2)
        
        # Should be different with high probability
        assert not np.array_equal(selected1, selected2)