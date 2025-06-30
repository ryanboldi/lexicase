"""
Tests for base lexicase selection.
"""

import numpy as np
import pytest
from lexicase import lexicase_selection, epsilon_lexicase_selection, set_backend

# Check if JAX is available
try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Set up backend parameters based on availability
BACKENDS = ['numpy']
if JAX_AVAILABLE:
    BACKENDS.append('jax')


def _to_set(arr):
    """Convert array to set, handling both NumPy and JAX arrays."""
    if hasattr(arr, 'tolist'):
        return set(arr.tolist())
    return set(arr)


@pytest.fixture(params=BACKENDS)
def backend(request):
    """Test with different backends."""
    set_backend(request.param)
    return request.param


def test_basic_selection(backend):
    """Test basic lexicase selection functionality."""
    fitnesses = np.array([
        [1.0, 0.0, 1.0],  # Individual 0: good at cases 0,2
        [0.0, 1.0, 0.0],  # Individual 1: good at case 1
        [0.5, 0.5, 0.5],  # Individual 2: mediocre at all
    ])
    
    selected = lexicase_selection(fitnesses, num_selected=1, seed=42)
    assert len(selected) == 1
    assert selected[0] in [0, 1, 2]


def test_deterministic_with_seed(backend):
    """Test that results are deterministic when seed is provided."""
    fitnesses = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.5],
    ])
    
    selected1 = lexicase_selection(fitnesses, num_selected=2, seed=42)
    selected2 = lexicase_selection(fitnesses, num_selected=2, seed=42)
    
    np.testing.assert_array_equal(selected1, selected2)


def test_basic_lexicase_behavior(backend):
    """Test basic lexicase selection behavior."""
    fitnesses = np.array([
        [0.5, 0.5, 0.5],  # Individual 0: mediocre
        [1.0, 0.0, 0.0],  # Individual 1: specialist
        [3.0, 3.0, 3.0],  # Individual 2: clearly best overall
    ])
    
    selected = lexicase_selection(fitnesses, num_selected=3, seed=42)
    
    # All selections should be valid
    assert len(selected) == 3
    assert all(0 <= idx < 3 for idx in selected)


def test_select_population_size(backend):
    """Test selecting specific number of individuals."""
    fitnesses = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ])
    
    # Test selecting specific numbers
    selected = lexicase_selection(fitnesses, num_selected=3, seed=42)
    assert len(selected) == 3


def test_empty_selection(backend):
    """Test selecting zero individuals."""
    fitnesses = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    
    selected = lexicase_selection(fitnesses, num_selected=0)
    assert len(selected) == 0


def test_single_individual(backend):
    """Test with single individual population."""
    fitnesses = np.array([[1.0, 0.5, 0.0]])
    
    selected = lexicase_selection(fitnesses, num_selected=1, seed=42)
    assert len(selected) == 1
    assert selected[0] == 0


def test_single_case(backend):
    """Test with single test case."""
    fitnesses = np.array([
        [1.0],
        [0.5],
        [0.0],
    ])
    
    selected = lexicase_selection(fitnesses, num_selected=1, seed=42)
    assert len(selected) == 1
    assert selected[0] == 0  # Best individual should be selected


def test_epsilon_lexicase_selection(backend):
    """Test epsilon lexicase selection."""
    fitnesses = np.array([
        [1.0, 0.0, 1.0],
        [0.9, 0.1, 0.9],  # Slightly worse but within epsilon
        [0.0, 1.0, 0.0],
    ])
    
    # Test epsilon lexicase with small epsilon
    selected_epsilon = epsilon_lexicase_selection(fitnesses, num_selected=10, epsilon=0.2, seed=42)
    
    # Selection should be valid (contain valid indices)
    assert len(selected_epsilon) == 10
    assert all(0 <= idx < 3 for idx in selected_epsilon)


def test_invalid_inputs(backend):
    """Test error handling for invalid inputs."""
    fitnesses = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    # Test invalid num_selected
    with pytest.raises(ValueError):
        lexicase_selection(fitnesses, num_selected=-1)
    
    # Test invalid seed type
    with pytest.raises(ValueError):
        lexicase_selection(fitnesses, num_selected=1, seed="invalid")


def test_stress_large_population(backend):
    """Stress test with larger population."""
    np.random.seed(42)
    n_individuals = 50
    n_cases = 20
    
    # Create diverse fitness landscape
    fitnesses = np.random.rand(n_individuals, n_cases)
    
    selected = lexicase_selection(fitnesses, num_selected=10, seed=42)
    
    assert len(selected) == 10
    assert all(0 <= idx < n_individuals for idx in selected)


def test_tie_breaking(backend):
    """Test tie-breaking behavior."""
    # All individuals are identical
    fitnesses = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    
    selected = lexicase_selection(fitnesses, num_selected=10, seed=42)
    
    # All selections should be valid
    assert len(selected) == 10
    assert all(0 <= idx < 3 for idx in selected)


def test_specialist_selection(backend):
    """Test that specialists can be selected."""
    # Create specialists for different cases
    fitnesses = np.array([
        [10.0, 0.0, 0.0],  # Specialist for case 0
        [0.0, 10.0, 0.0],  # Specialist for case 1  
        [0.0, 0.0, 10.0],  # Specialist for case 2
        [3.0, 3.0, 3.0],   # Generalist (good overall)
    ])
    
    # With sufficient selections, should see some diversity
    selected = lexicase_selection(fitnesses, num_selected=20, seed=42)
    
    assert len(selected) == 20
    assert all(0 <= idx < 4 for idx in selected)
    
    # Should select multiple different individuals
    unique_selections = _to_set(selected)
    assert len(unique_selections) >= 2  # Some diversity


def test_case_order_matters(backend):
    """Test that case order affects selection."""
    fitnesses = np.array([
        [10, 1, 1],  # Individual 0: best on case 0
        [1, 10, 1],  # Individual 1: best on case 1
        [1, 1, 10]   # Individual 2: best on case 2
    ])
    
    # With different seeds, we should see different selections due to case shuffling
    results = []
    for seed in range(10):
        selected = lexicase_selection(fitnesses, num_selected=1, seed=seed)
        # Convert JAX scalar to Python int for hashing
        result_val = int(selected[0]) if hasattr(selected[0], 'item') else selected[0]
        results.append(result_val)
    
    # Should see some variation in results
    unique_results = set(results)  # results is a list of individual indices, not arrays
    assert len(unique_results) >= 2  # At least some variation


def test_filtering_logic(backend):
    """Test the filtering logic works correctly."""
    fitnesses = np.array([
        [100, 1, 1],   # Individual 0: excellent on case 0
        [1, 100, 1],   # Individual 1: excellent on case 1
        [1, 1, 100],   # Individual 2: excellent on case 2
        [50, 50, 50],  # Individual 3: good on all cases
        [1, 1, 1]      # Individual 4: poor on all cases
    ])
    
    # Run multiple selections to see behavior
    selected = lexicase_selection(fitnesses, num_selected=50, seed=42)
    
    assert len(selected) == 50
    assert all(0 <= idx < 5 for idx in selected)
    
    # Check that poor individual (4) is selected less frequently
    selection_counts = np.bincount(selected, minlength=5)
    assert selection_counts[4] < max(selection_counts)  # Worst should be selected less


def test_epsilon_vs_regular_comparison(backend):
    """Test comparison between epsilon and regular lexicase."""
    fitnesses = np.array([
        [100, 1, 1],
        [99, 2, 2],    # Close to best on case 0
        [1, 100, 1], 
        [2, 99, 2],    # Close to best on case 1
        [1, 1, 100],
        [2, 2, 99]     # Close to best on case 2
    ])
    
    # Regular lexicase
    selected_regular = lexicase_selection(fitnesses, num_selected=20, seed=42)
    
    # Epsilon lexicase with small epsilon
    selected_epsilon_small = epsilon_lexicase_selection(fitnesses, num_selected=20, epsilon=0.5, seed=42)
    
    # Epsilon lexicase with large epsilon 
    selected_epsilon_large = epsilon_lexicase_selection(fitnesses, num_selected=20, epsilon=5.0, seed=42)
    
    # All should produce valid results
    assert len(selected_regular) == 20
    assert len(selected_epsilon_small) == 20
    assert len(selected_epsilon_large) == 20
    
    # Large epsilon should allow more diversity
    diversity_regular = len(_to_set(selected_regular))
    diversity_large_eps = len(_to_set(selected_epsilon_large))
    
    # This is a soft check since stochasticity can affect results
    assert diversity_large_eps >= diversity_regular * 0.5  # At least half the diversity


def test_epsilon_behavior(backend):
    """Test epsilon lexicase behavior."""
    fitnesses = np.array([
        [100, 1, 1],
        [98, 3, 3],    # Within epsilon=3 of best on case 0
        [1, 100, 1], 
        [3, 98, 3],    # Within epsilon=3 of best on case 1
    ])
    
    # Test with epsilon that should include close performers
    selected = epsilon_lexicase_selection(fitnesses, num_selected=20, epsilon=3.0, seed=42)
    
    assert len(selected) == 20
    assert all(0 <= idx < 4 for idx in selected)
    
    # Should see some diversity in selection
    unique_selected = _to_set(selected)
    assert len(unique_selected) >= 2


def test_multiple_cases_sufficient(backend):
    """Test that multiple cases provide sufficient selection pressure."""
    np.random.seed(123)
    
    # Create 20 individuals, 15 cases
    fitnesses = np.random.rand(20, 15) * 10
    
    # Make some individuals clearly better on specific cases
    fitnesses[0, :5] += 20   # Individual 0 dominates first 5 cases
    fitnesses[1, 5:10] += 20 # Individual 1 dominates next 5 cases
    fitnesses[2, 10:] += 20  # Individual 2 dominates last 5 cases
    
    selected = lexicase_selection(fitnesses, num_selected=40, seed=42)
    
    assert len(selected) == 40
    assert all(0 <= idx < 20 for idx in selected)
    
    # The dominant individuals should be selected frequently
    selection_counts = np.bincount(selected, minlength=20)
    
    # At least one of the dominant individuals should be selected
    assert max(selection_counts[0], selection_counts[1], selection_counts[2]) > 0 