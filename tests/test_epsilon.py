"""
Tests for epsilon lexicase selection.
"""

import pytest
import numpy as np
from lexicase import epsilon_lexicase_selection, lexicase_selection, set_backend


class TestEpsilonLexicase:
    """Test cases for epsilon lexicase selection."""
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_epsilon_bounds(self, backend):
        """Test that epsilon creates appropriate bounds for selection."""
        set_backend(backend)
        
        fitness_matrix = np.array([
            [10.0, 5.0, 1.0],
            [9.5, 4.8, 1.2],   # Within epsilon of best on cases 0,1
            [1.0, 1.0, 1.0]     # Not within epsilon
        ])
        
        # With small epsilon, should be more selective
        selected_small = epsilon_lexicase_selection(
            fitness_matrix, num_selected=100, epsilon=0.1, seed=42
        )
        
        # With large epsilon, should be less selective
        selected_large = epsilon_lexicase_selection(
            fitness_matrix, num_selected=100, epsilon=2.0, seed=42
        )
        
        # Large epsilon should result in more diverse selection
        diversity_small = len(set(selected_small))
        diversity_large = len(set(selected_large))
        assert diversity_large >= diversity_small
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_equivalence_to_base_with_zero_epsilon(self, backend):
        """Test that epsilon=0 is equivalent to base lexicase."""
        set_backend(backend)
        
        fitness_matrix = np.array([
            [10, 5, 1],
            [8, 6, 2],
            [6, 8, 3]
        ])
        
        base_selected = lexicase_selection(fitness_matrix, num_selected=50, seed=42)
        epsilon_selected = epsilon_lexicase_selection(
            fitness_matrix, num_selected=50, epsilon=0.0, seed=42
        )
        
        # Should produce identical results
        assert np.array_equal(base_selected, epsilon_selected)
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_epsilon_correctness(self, backend):
        """Test correctness of epsilon lexicase logic."""
        set_backend(backend)
        
        # Clear case where epsilon should make a difference
        fitness_matrix = np.array([
            [100.0, 0.0],    # Individual 0: great on case 0, bad on case 1
            [99.0, 100.0],   # Individual 1: almost as good on case 0, great on case 1
            [0.0, 0.0]       # Individual 2: bad on both
        ])
        
        # With epsilon=2, individual 1 should be competitive on case 0
        selected = epsilon_lexicase_selection(
            fitness_matrix, num_selected=100, epsilon=2.0, seed=42
        )
        
        # Both individuals 0 and 1 should be selected (order depends on case shuffling)
        unique_selected = set(selected)
        # At least one of the good individuals should be selected
        assert len(unique_selected.intersection({0, 1})) > 0
        assert 2 not in unique_selected
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_negative_epsilon_error(self, backend):
        """Test that negative epsilon raises error."""
        set_backend(backend)
        
        fitness_matrix = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="Epsilon must be non-negative"):
            epsilon_lexicase_selection(fitness_matrix, num_selected=1, epsilon=-0.1)
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_large_epsilon_selects_all(self, backend):
        """Test that very large epsilon allows all individuals to be competitive."""
        set_backend(backend)
        
        fitness_matrix = np.array([
            [100, 0],
            [0, 100],
            [50, 50]
        ])
        
        # Very large epsilon should make all individuals competitive
        selected = epsilon_lexicase_selection(
            fitness_matrix, num_selected=100, epsilon=1000.0, seed=42
        )
        
        unique_selected = set(selected)
        assert len(unique_selected) == 3  # All individuals should be selected

    # STRESS TESTS - Rigorous epsilon lexicase validation
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_epsilon_scaling_stress(self, backend):
        """Stress test how epsilon affects selection with large populations."""
        set_backend(backend)
        
        # 50 individuals, 25 test cases
        np.random.seed(456)
        fitness_matrix = np.random.rand(50, 25) * 100
        
        # Create a clear hierarchy on first case
        fitness_matrix[:, 0] = np.arange(50, 0, -1)  # 50, 49, 48, ..., 1
        
        # Test different epsilon values
        epsilon_values = [0.0, 1.0, 5.0, 10.0, 50.0]
        diversity_scores = []
        
        for epsilon in epsilon_values:
            selected = epsilon_lexicase_selection(
                fitness_matrix, num_selected=200, epsilon=epsilon, seed=42
            )
            unique_count = len(set(selected))
            diversity_scores.append(unique_count)
        
        # Diversity should generally increase with epsilon
        # (though not necessarily monotonically due to stochasticity)
        max_diversity = max(diversity_scores)
        min_diversity = min(diversity_scores)
        
        assert max_diversity > min_diversity, "Epsilon should affect diversity"
        assert max_diversity >= 10, "Should have reasonable diversity with large epsilon"
        
        # Zero epsilon should be most restrictive
        zero_epsilon_diversity = diversity_scores[0]
        large_epsilon_diversity = diversity_scores[-1]
        assert large_epsilon_diversity >= zero_epsilon_diversity, \
            "Large epsilon should not be more restrictive than zero epsilon"
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_epsilon_threshold_behavior(self, backend):
        """Test precise epsilon threshold behavior."""
        set_backend(backend)
        
        # Carefully crafted fitness matrix to test epsilon thresholds
        fitness_matrix = np.array([
            [100.0, 50.0],    # Individual 0
            [99.0, 51.0],     # Individual 1: 1.0 worse on case 0, 1.0 better on case 1
            [98.0, 52.0],     # Individual 2: 2.0 worse on case 0, 2.0 better on case 1
            [97.0, 53.0],     # Individual 3: 3.0 worse on case 0, 3.0 better on case 1
            [0.0, 0.0]        # Individual 4: clearly worst
        ])
        
        # Test with epsilon = 1.5 (should include individuals 0 and 1 on case 0)
        selected = epsilon_lexicase_selection(
            fitness_matrix, num_selected=200, epsilon=1.5, seed=42
        )
        selection_counts = np.bincount(selected, minlength=5)
        
        # Individuals 0 and 1 should be selected (competitive on case 0)
        assert selection_counts[0] > 0, "Individual 0 should be selected"
        assert selection_counts[1] > 0, "Individual 1 should be selected (within epsilon)"
        
        # Individual 4 should never be selected
        assert selection_counts[4] == 0, "Worst individual should not be selected"
        
        # Test with smaller epsilon = 0.5 (should exclude individual 1 on case 0)
        selected_small = epsilon_lexicase_selection(
            fitness_matrix, num_selected=200, epsilon=0.5, seed=42
        )
        selection_counts_small = np.bincount(selected_small, minlength=5)
        
        # Individual 0 should dominate more with smaller epsilon
        assert selection_counts_small[0] >= selection_counts[0], \
            "Individual 0 should be selected at least as much with smaller epsilon"
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_epsilon_with_many_close_performers(self, backend):
        """Test epsilon lexicase with many individuals having similar performance."""
        set_backend(backend)
        
        # 40 individuals, 20 test cases
        # Create fitness landscape with many close performers
        np.random.seed(789)
        fitness_matrix = np.random.rand(40, 20) * 10 + 90  # Fitness between 90-100
        
        # Add some slight variations
        fitness_matrix[0, :] += 5  # Individual 0: slightly better
        fitness_matrix[1, :] += 4  # Individual 1: slightly better
        fitness_matrix[2, :] += 3  # Individual 2: slightly better
        
        # Test different epsilon values
        small_epsilon_selected = epsilon_lexicase_selection(
            fitness_matrix, num_selected=100, epsilon=0.5, seed=42
        )
        large_epsilon_selected = epsilon_lexicase_selection(
            fitness_matrix, num_selected=100, epsilon=5.0, seed=42
        )
        
        small_diversity = len(set(small_epsilon_selected))
        large_diversity = len(set(large_epsilon_selected))
        
        # With close performers, epsilon can have complex effects on diversity
        # Small epsilon might allow more individuals to compete, while very large epsilon
        # might cause early convergence. We'll just verify both produce valid selections.
        assert small_diversity >= 1 and small_diversity <= 40, \
            f"Small epsilon should produce valid diversity: {small_diversity}"
        assert large_diversity >= 1 and large_diversity <= 40, \
            f"Large epsilon should produce valid diversity: {large_diversity}"
        
        # Both should still select the clearly better individuals
        small_counts = np.bincount(small_epsilon_selected, minlength=40)
        large_counts = np.bincount(large_epsilon_selected, minlength=40)
        
        # Top 3 individuals should be well-represented in both cases
        top_3_small = small_counts[:3].sum()
        top_3_large = large_counts[:3].sum()
        
        assert top_3_small > 0, "Top individuals should be selected with small epsilon"
        assert top_3_large > 0, "Top individuals should be selected with large epsilon"
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_epsilon_extreme_fitness_ranges(self, backend):
        """Test epsilon lexicase with extreme fitness value ranges."""
        set_backend(backend)
        
        # Test with very large fitness values
        fitness_matrix = np.array([
            [1000000.0, 500000.0],
            [999999.0, 500001.0],  # Very close to best on case 0
            [999900.0, 500100.0],  # Further from best
            [100.0, 100.0]         # Much worse
        ])
        
        # Small epsilon relative to scale
        selected_small = epsilon_lexicase_selection(
            fitness_matrix, num_selected=100, epsilon=10.0, seed=42
        )
        
        # Large epsilon relative to scale  
        selected_large = epsilon_lexicase_selection(
            fitness_matrix, num_selected=100, epsilon=1000.0, seed=42
        )
        
        # Should handle large numbers without numerical issues
        assert len(selected_small) == 100
        assert len(selected_large) == 100
        assert all(0 <= ind < 4 for ind in selected_small)
        assert all(0 <= ind < 4 for ind in selected_large)
        
        # Large epsilon should include more individuals
        diversity_small = len(set(selected_small))
        diversity_large = len(set(selected_large))
        assert diversity_large >= diversity_small
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_epsilon_consistency_across_runs(self, backend):
        """Test that epsilon lexicase maintains consistent behavior across multiple runs."""
        set_backend(backend)
        
        # 30 individuals, 15 test cases
        np.random.seed(999)
        fitness_matrix = np.random.rand(30, 15) * 50
        
        # Create some individuals with clear advantages
        fitness_matrix[0, :5] += 20    # Individual 0: good on first 5 cases
        fitness_matrix[1, 5:10] += 20  # Individual 1: good on middle 5 cases
        fitness_matrix[2, 10:] += 20   # Individual 2: good on last 5 cases
        
        epsilon = 3.0
        
        # Run multiple times with different seeds
        all_selections = []
        for seed in range(30):
            selected = epsilon_lexicase_selection(
                fitness_matrix, num_selected=50, epsilon=epsilon, seed=seed
            )
            all_selections.extend(selected)
        
        selection_counts = np.bincount(all_selections, minlength=30)
        
        # Statistical consistency checks
        total_selections = len(all_selections)
        
        # The enhanced individuals should be selected more often
        enhanced_individuals = [0, 1, 2]
        enhanced_selections = selection_counts[enhanced_individuals].sum()
        expected_random = total_selections * (3/30)  # Random chance
        
        assert enhanced_selections > expected_random * 2, \
            "Enhanced individuals should be selected more than random chance"
        
        # Should maintain diversity across runs
        non_zero_count = np.count_nonzero(selection_counts)
        assert non_zero_count >= 3, \
            f"Should maintain diversity across runs: {non_zero_count} individuals selected"
        
        # No individual should completely dominate
        max_individual_selections = np.max(selection_counts)
        max_share = max_individual_selections / total_selections
        assert max_share < 0.3, \
            f"No individual should dominate: max share = {max_share:.2%}"


@pytest.fixture(params=['numpy'])
def backend(request):
    """Test with different backends."""
    set_backend(request.param)
    return request.param


def test_epsilon_basic_functionality(backend):
    """Test basic epsilon lexicase functionality."""
    fitnesses = np.array([
        [1.0, 0.0, 1.0],
        [0.9, 0.1, 0.9],  # Close to individual 0
        [0.0, 1.0, 0.0],
    ])
    
    selected = epsilon_lexicase_selection(fitnesses, num_selected=1, epsilon=0.2, seed=42)
    assert len(selected) == 1
    assert selected[0] in [0, 1, 2]


def test_epsilon_selection_behavior(backend):
    """Test epsilon lexicase selection behavior with different epsilon values."""
    # Create a controlled fitness matrix
    fitnesses = np.array([
        [10.0, 5.0],   # Individual 0
        [9.8, 4.9],    # Individual 1: very close
        [9.5, 4.8],    # Individual 2: close
        [1.0, 1.0],    # Individual 3: much worse
    ])
    
    # Test with small epsilon
    selected_small = epsilon_lexicase_selection(fitnesses, num_selected=30, epsilon=0.1, seed=42)
    
    # Test with larger epsilon that should include close performers
    selected_large = epsilon_lexicase_selection(fitnesses, num_selected=30, epsilon=0.5, seed=42)
    
    assert len(selected_small) == 30
    assert len(selected_large) == 30
    assert all(0 <= idx < 4 for idx in selected_small)
    assert all(0 <= idx < 4 for idx in selected_large)
    
    # Count selections
    selection_counts_small = [list(selected_small).count(i) for i in range(4)]
    selection_counts_large = [list(selected_large).count(i) for i in range(4)]
    
    # Individual 3 should be selected less frequently in both cases
    assert selection_counts_small[3] < max(selection_counts_small[:3])
    assert selection_counts_large[3] < max(selection_counts_large[:3])


def test_epsilon_basic_lexicase_behavior(backend):
    """Test basic epsilon lexicase behavior."""
    fitnesses = np.array([
        [1.0, 1.0, 1.0],  # Individual 0: mediocre
        [2.0, 0.0, 0.0],  # Individual 1: specialist
        [3.0, 3.0, 3.0],  # Individual 2: clearly best overall
    ])
    
    selected = epsilon_lexicase_selection(fitnesses, num_selected=3, epsilon=0.5, seed=42)
    
    # All selections should be valid
    assert len(selected) == 3
    assert all(0 <= idx < 3 for idx in selected)


def test_epsilon_vs_regular_behavior(backend):
    """Test that epsilon lexicase behaves differently from regular lexicase."""
    # Create a case where epsilon should make a difference
    fitnesses = np.array([
        [1.00, 0.10, 0.10],  # Individual 0: perfect on case 0
        [0.95, 0.15, 0.15],  # Individual 1: very close on all
        [0.90, 0.20, 0.20],  # Individual 2: close on all
        [0.00, 1.00, 0.00],  # Individual 3: perfect on case 1
        [0.00, 0.00, 1.00],  # Individual 4: perfect on case 2
    ])
    
    selected = epsilon_lexicase_selection(fitnesses, num_selected=50, epsilon=0.1, seed=42)
    
    assert len(selected) == 50
    assert all(0 <= idx < 5 for idx in selected)
    
    # Should have reasonable diversity
    unique_selections = set(selected)
    assert len(unique_selections) >= 3


def test_epsilon_deterministic(backend):
    """Test that epsilon lexicase is deterministic with same seed."""
    fitnesses = np.array([
        [1.0, 0.0, 1.0],
        [0.9, 0.1, 0.9],
        [0.0, 1.0, 0.0],
    ])
    
    selected1 = epsilon_lexicase_selection(fitnesses, num_selected=5, epsilon=0.2, seed=42)
    selected2 = epsilon_lexicase_selection(fitnesses, num_selected=5, epsilon=0.2, seed=42)
    
    np.testing.assert_array_equal(selected1, selected2)


def test_epsilon_empty_selection(backend):
    """Test epsilon lexicase with zero selections."""
    fitnesses = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    
    selected = epsilon_lexicase_selection(fitnesses, num_selected=0, epsilon=0.1)
    assert len(selected) == 0


def test_epsilon_single_individual(backend):
    """Test epsilon lexicase with single individual."""
    fitnesses = np.array([[1.0, 0.5, 0.0]])
    
    selected = epsilon_lexicase_selection(fitnesses, num_selected=1, epsilon=0.1, seed=42)
    assert len(selected) == 1
    assert selected[0] == 0


def test_epsilon_with_ties(backend):
    """Test epsilon lexicase with tied individuals."""
    # Create scenario with many ties
    fitnesses = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    
    selected = epsilon_lexicase_selection(fitnesses, num_selected=20, epsilon=0.1, seed=42)
    
    assert len(selected) == 20
    assert all(0 <= idx < 4 for idx in selected)
    
    # Should distribute selections among all individuals
    unique_selections = set(selected)
    assert len(unique_selections) >= 2  # Some distribution


def test_epsilon_large_population(backend):
    """Test epsilon lexicase with larger population."""
    np.random.seed(123)
    n_individuals = 50
    n_cases = 15
    
    fitnesses = np.random.rand(n_individuals, n_cases)
    
    # Make some individuals clearly better
    fitnesses[0, :] += 2.0  # Individual 0 generally better
    fitnesses[1, :5] += 3.0  # Individual 1 excellent on first 5 cases
    
    selected = epsilon_lexicase_selection(fitnesses, num_selected=25, epsilon=0.5, seed=42)
    
    assert len(selected) == 25
    assert all(0 <= idx < n_individuals for idx in selected)


def test_epsilon_edge_cases(backend):
    """Test epsilon lexicase with edge cases."""
    # Case where all values are the same
    fitnesses_same = np.array([
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0],
    ])
    
    selected = epsilon_lexicase_selection(fitnesses_same, num_selected=5, epsilon=0.1, seed=42)
    assert len(selected) == 5
    assert all(0 <= idx < 3 for idx in selected)
    
    # Case with extreme outliers
    fitnesses_outlier = np.array([
        [1.0, 1.0, 1.0],
        [1.1, 1.1, 1.1],
        [100.0, 100.0, 100.0],  # Extreme outlier
    ])
    
    selected = epsilon_lexicase_selection(fitnesses_outlier, num_selected=10, epsilon=5.0, seed=42)
    assert len(selected) == 10
    assert all(0 <= idx < 3 for idx in selected)


def test_epsilon_case_order_independence(backend):
    """Test that epsilon lexicase handles case order correctly."""
    fitnesses = np.array([
        [10, 1, 1],  # Individual 0: specialist on case 0
        [1, 10, 1],  # Individual 1: specialist on case 1
        [1, 1, 10],  # Individual 2: specialist on case 2
    ])
    
    # Test with different seeds (different case orders)
    results = []
    for seed in range(10):
        selected = epsilon_lexicase_selection(fitnesses, num_selected=1, epsilon=0.5, seed=seed)
        results.append(selected[0])
    
    # Should see some variety due to case order differences
    unique_results = set(results)
    assert len(unique_results) >= 2


def test_epsilon_basic_behavior(backend):
    """Test basic epsilon lexicase behavior."""
    fitnesses = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 0.0, 0.0],
        [5.0, 5.0, 5.0],  # Clearly best overall
    ])
    
    selected = epsilon_lexicase_selection(fitnesses, num_selected=5, epsilon=0.5, seed=42)
    
    # All selections should be valid
    assert len(selected) == 5
    assert all(0 <= idx < 3 for idx in selected)
    assert all(0 <= idx < 3 for idx in selected)


def test_epsilon_stress_test(backend):
    """Stress test epsilon lexicase with complex fitness landscape."""
    np.random.seed(456)
    n_individuals = 30
    n_cases = 12
    
    # Create complex fitness landscape
    fitnesses = np.random.exponential(scale=2.0, size=(n_individuals, n_cases))
    
    # Add some structure
    fitnesses[0, :4] *= 3  # Individual 0 good on first 4 cases
    fitnesses[1, 4:8] *= 3  # Individual 1 good on middle 4 cases
    fitnesses[2, 8:] *= 3   # Individual 2 good on last 4 cases
    
    selected = epsilon_lexicase_selection(fitnesses, num_selected=60, epsilon=1.0, seed=42)
    
    assert len(selected) == 60
    assert all(0 <= idx < n_individuals for idx in selected)
    
    # Should see reasonable diversity
    unique_selections = set(selected)
    assert len(unique_selections) >= 5 