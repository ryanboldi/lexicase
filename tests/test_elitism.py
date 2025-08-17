"""
Tests for elitism parameter in lexicase selection functions.
"""

import numpy as np
import pytest
from lexicase import (
    lexicase_selection,
    epsilon_lexicase_selection,
    downsample_lexicase_selection
)

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class TestElitism:
    """Test elitism parameter functionality."""
    
    def test_lexicase_elitism_numpy(self):
        """Test that elitism selects best individuals in NumPy."""
        # Create fitness matrix where individuals 8 and 9 are clearly best
        fitness_matrix = np.array([
            [1, 2, 1, 2, 1],  # Total: 7
            [2, 1, 2, 1, 2],  # Total: 8
            [1, 1, 1, 1, 1],  # Total: 5
            [2, 2, 1, 1, 1],  # Total: 7
            [1, 1, 2, 2, 1],  # Total: 7
            [2, 1, 1, 1, 2],  # Total: 7
            [1, 2, 2, 1, 1],  # Total: 7
            [2, 2, 2, 1, 1],  # Total: 8
            [3, 3, 3, 3, 3],  # Total: 15 - Best
            [3, 3, 3, 3, 2],  # Total: 14 - Second best
        ])
        
        # Select 5 individuals with elitism=2
        selected = lexicase_selection(fitness_matrix, 5, seed=42, elitism=2)
        
        # Check that the top 2 individuals (8 and 9) are always selected
        assert 8 in selected
        assert 9 in selected
        assert len(selected) == 5
        
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_lexicase_elitism_jax(self):
        """Test that elitism selects best individuals in JAX."""
        # Same test as NumPy but with JAX arrays
        fitness_matrix = jnp.array([
            [1, 2, 1, 2, 1],  # Total: 7
            [2, 1, 2, 1, 2],  # Total: 8
            [1, 1, 1, 1, 1],  # Total: 5
            [2, 2, 1, 1, 1],  # Total: 7
            [1, 1, 2, 2, 1],  # Total: 7
            [2, 1, 1, 1, 2],  # Total: 7
            [1, 2, 2, 1, 1],  # Total: 7
            [2, 2, 2, 1, 1],  # Total: 8
            [3, 3, 3, 3, 3],  # Total: 15 - Best
            [3, 3, 3, 3, 2],  # Total: 14 - Second best
        ])
        
        # Select 5 individuals with elitism=2
        selected = lexicase_selection(fitness_matrix, 5, seed=42, elitism=2)
        
        # Check that the top 2 individuals (8 and 9) are always selected
        assert 8 in selected
        assert 9 in selected
        assert len(selected) == 5
        
    def test_epsilon_lexicase_elitism_numpy(self):
        """Test elitism in epsilon lexicase selection."""
        fitness_matrix = np.array([
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 2.0],
            [5.0, 5.0, 5.0],  # Best individual
            [1.0, 1.0, 1.0],
            [4.0, 4.0, 4.0],  # Second best
        ])
        
        # Select 3 with elitism=2
        selected = epsilon_lexicase_selection(
            fitness_matrix, 3, epsilon=0.5, seed=42, elitism=2
        )
        
        # Top 2 should always be selected
        assert 2 in selected  # Index 2 has total fitness 15
        assert 4 in selected  # Index 4 has total fitness 12
        assert len(selected) == 3
        
    def test_downsample_lexicase_elitism_numpy(self):
        """Test elitism in downsampled lexicase selection."""
        fitness_matrix = np.array([
            [1, 2, 1, 2, 1, 1],
            [2, 1, 2, 1, 2, 1],
            [1, 1, 1, 1, 1, 1],
            [5, 5, 5, 5, 5, 5],  # Best
            [4, 4, 4, 4, 4, 4],  # Second best
        ])
        
        # Select 3 with elitism=1 and downsample_size=3
        selected = downsample_lexicase_selection(
            fitness_matrix, 3, downsample_size=3, seed=42, elitism=1
        )
        
        # Best individual should always be selected
        assert 3 in selected  # Index 3 has highest total fitness
        assert len(selected) == 3
        
    def test_elitism_edge_cases(self):
        """Test edge cases for elitism parameter."""
        fitness_matrix = np.array([
            [1, 2, 3],
            [2, 3, 1],
            [3, 1, 2],
        ])
        
        # Test elitism=0 (no elitism)
        selected = lexicase_selection(fitness_matrix, 2, seed=42, elitism=0)
        assert len(selected) == 2
        
        # Test elitism=num_selected (all elite)
        selected = lexicase_selection(fitness_matrix, 3, seed=42, elitism=3)
        assert len(selected) == 3
        # All individuals should be selected based on total fitness ranking
        assert set(selected) == {0, 1, 2}
        
    def test_elitism_validation(self):
        """Test that invalid elitism values raise errors."""
        fitness_matrix = np.array([[1, 2], [2, 1]])
        
        # Negative elitism
        with pytest.raises(ValueError, match="Elitism must be non-negative"):
            lexicase_selection(fitness_matrix, 2, elitism=-1)
            
        # Elitism > num_selected
        with pytest.raises(ValueError, match="Elitism cannot exceed num_selected"):
            lexicase_selection(fitness_matrix, 1, elitism=2)
            
        # Elitism > number of individuals (need higher num_selected to test this)
        with pytest.raises(ValueError, match="Elitism cannot exceed number of individuals"):
            lexicase_selection(fitness_matrix, 10, elitism=3)
            
    def test_elitism_deterministic(self):
        """Test that elite selection is deterministic."""
        fitness_matrix = np.array([
            [1, 1, 1],  # Total: 3
            [2, 2, 2],  # Total: 6
            [3, 3, 3],  # Total: 9
            [4, 4, 4],  # Total: 12
            [5, 5, 5],  # Total: 15
        ])
        
        # Run multiple times with different seeds but same elitism
        results = []
        for seed in range(10):
            selected = lexicase_selection(fitness_matrix, 3, seed=seed, elitism=2)
            # Extract elite individuals (first 2 in selection)
            elite = sorted(selected[:2])
            results.append(elite)
            
        # All runs should have the same elite individuals (3 and 4)
        for elite in results:
            assert elite == [3, 4]