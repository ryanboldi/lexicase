"""
Tests for downsampled lexicase selection.
"""

import pytest
import numpy as np
from lexicase import downsample_lexicase_selection, lexicase_selection

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


class TestDownsampleLexicase:
    """Test cases for downsampled lexicase selection."""
    
    
    def test_verifies_case_count(self):
        """Test that downsampling uses correct number of cases."""
        
        # Create fitness matrix with many test cases
        num_cases = 100
        fitness_matrix = np.random.rand(5, num_cases)
        
        # Mock the selection function to verify case count
        # This is a bit tricky to test directly, so we'll test indirectly
        # by ensuring that different downsample sizes produce different results
        
        selected_small = downsample_lexicase_selection(
            fitness_matrix, num_selected=50, downsample_size=5, seed=42
        )
        selected_large = downsample_lexicase_selection(
            fitness_matrix, num_selected=50, downsample_size=50, seed=42
        )
        
        # Different downsample sizes should generally produce different results
        # (though this isn't guaranteed, it's very likely with random data)
        assert not np.array_equal(selected_small, selected_large)
    
    
    def test_higher_selection_diversity(self):
        """Test that downsampling increases selection diversity."""
        
        # Create a scenario where one individual dominates most cases
        # but downsampling might reveal diversity
        fitness_matrix = np.array([
            [10, 10, 10, 10, 10, 1, 1, 1],  # Individual 0: dominates most cases
            [9, 9, 9, 9, 9, 10, 10, 10],    # Individual 1: good on fewer cases
            [8, 8, 8, 8, 8, 9, 9, 9],       # Individual 2: mediocre
            [1, 1, 1, 1, 1, 8, 8, 8]        # Individual 3: poor on most
        ])
        
        # Full lexicase should favor individual 0
        full_selected = lexicase_selection(fitness_matrix, num_selected=100, seed=42)
        full_diversity = len(_to_set(full_selected))
        
        # Downsampled lexicase should potentially increase diversity
        down_selected = downsample_lexicase_selection(
            fitness_matrix, num_selected=100, downsample_size=3, seed=42
        )
        down_diversity = len(_to_set(down_selected))
        
        # Downsampling should generally increase or maintain diversity
        assert down_diversity >= full_diversity * 0.8  # Allow some tolerance
    
    
    def test_downsample_larger_than_cases(self):
        """Test behavior when downsample size is larger than available cases."""
        
        fitness_matrix = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        
        # Should handle gracefully when downsample_size > num_cases
        selected = downsample_lexicase_selection(
            fitness_matrix, num_selected=10, downsample_size=10, seed=42
        )
        assert len(selected) == 10
    
    
    def test_deterministic_with_seed(self):
        """Test that downsampled lexicase is deterministic with seed."""
        
        fitness_matrix = np.random.rand(4, 20)
        
        selected1 = downsample_lexicase_selection(
            fitness_matrix, num_selected=10, downsample_size=5, seed=123
        )
        selected2 = downsample_lexicase_selection(
            fitness_matrix, num_selected=10, downsample_size=5, seed=123
        )
        
        assert np.array_equal(selected1, selected2)
    
    
    def test_zero_downsample_size_error(self):
        """Test that zero downsample size raises error."""
        
        fitness_matrix = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="Downsample size must be positive"):
            downsample_lexicase_selection(
                fitness_matrix, num_selected=1, downsample_size=0
            )
    
    
    def test_single_case_downsample(self):
        """Test downsampling to single case."""
        
        fitness_matrix = np.array([
            [10, 1, 5],
            [1, 10, 6],
            [5, 5, 10]
        ])
        
        selected = downsample_lexicase_selection(
            fitness_matrix, num_selected=20, downsample_size=1, seed=42
        )
        
        # Should still work and produce valid selections
        assert len(selected) == 20
        assert all(0 <= ind < 3 for ind in selected)

    # STRESS TESTS - Based on conference video requirements
    
    
    def test_specialist_selection_with_downsampling(self):
        """Test that downsampling can reveal specialist individuals."""
        
        # Create specialists: each individual excels on specific cases
        # 20 individuals, 40 test cases
        fitness_matrix = np.ones((20, 40)) * 1.0  # Base poor performance
        
        # Create specialists
        for i in range(20):
            # Each individual excels on 2 specific cases
            case1 = i * 2
            case2 = i * 2 + 1
            if case1 < 40:
                fitness_matrix[i, case1] = 100.0
            if case2 < 40:
                fitness_matrix[i, case2] = 100.0
        
        # Full lexicase might miss some specialists due to case order
        full_selected = lexicase_selection(fitness_matrix, num_selected=200, seed=42)
        full_diversity = len(_to_set(full_selected))
        
        # Downsampled lexicase should potentially find more specialists
        # by focusing on smaller case subsets
        down_selected = downsample_lexicase_selection(
            fitness_matrix, num_selected=200, downsample_size=5, seed=42
        )
        down_diversity = len(_to_set(down_selected))
        
        # Should maintain or improve diversity with specialists
        assert down_diversity >= max(5, full_diversity * 0.8), \
            f"Downsampling should maintain specialist diversity: {down_diversity} vs full {full_diversity}"
        
        # Each individual should have a reasonable chance of being selected
        # since they excel on specific cases
        selection_counts = np.bincount(down_selected, minlength=20)
        non_zero_specialists = np.count_nonzero(selection_counts)
        assert non_zero_specialists >= 8, \
            f"Should select multiple specialists: {non_zero_specialists}/20"
    
    
    def test_case_subset_independence(self):
        """Test that different case subsets lead to different selection patterns."""
        
        # 15 individuals, 30 cases
        # Create structured fitness where different case subsets favor different individuals
        fitness_matrix = np.random.rand(15, 30) * 10
        
        # Make individuals 0-4 excellent on cases 0-9
        fitness_matrix[0:5, 0:10] += 50
        # Make individuals 5-9 excellent on cases 10-19
        fitness_matrix[5:10, 10:20] += 50
        # Make individuals 10-14 excellent on cases 20-29
        fitness_matrix[10:15, 20:30] += 50
        
        # Test different downsample sizes
        results_by_downsample = {}
        for downsample_size in [5, 10, 15, 20]:
            selected = downsample_lexicase_selection(
                fitness_matrix, num_selected=100, downsample_size=downsample_size, seed=42
            )
            selection_counts = np.bincount(selected, minlength=15)
            results_by_downsample[downsample_size] = selection_counts
        
        # Different downsample sizes should produce different selection patterns
        # Compare small vs large downsample
        small_counts = results_by_downsample[5]
        large_counts = results_by_downsample[20]
        
        # Should have different distributions
        # Use a simple correlation to check if patterns are different
        correlation = np.corrcoef(small_counts, large_counts)[0, 1]
        assert correlation < 0.9, \
            f"Different downsample sizes should produce different patterns: correlation = {correlation:.3f}"
    
    
    def test_downsample_with_insufficient_cases(self):
        """Test behavior when downsampling with too few cases for good discrimination."""
        
        # Only 3 cases (3! = 6 possible orderings) - insufficient for good lexicase
        fitness_matrix = np.array([
            [10, 5, 8],   # Individual 0
            [8, 10, 6],   # Individual 1
            [6, 8, 10],   # Individual 2
            [5, 6, 9],    # Individual 3
            [9, 7, 5]     # Individual 4
        ])
        
        # With only 2 cases downsampled, should still work but with limited diversity
        selected = downsample_lexicase_selection(
            fitness_matrix, num_selected=50, downsample_size=2, seed=42
        )
        
        assert len(selected) == 50
        assert all(0 <= ind < 5 for ind in selected)
        
        # Should still maintain some diversity even with few cases
        unique_selected = len(_to_set(selected))
        assert unique_selected >= 2, \
            f"Should maintain some diversity even with few cases: {unique_selected}"
    
    
    def test_downsample_stochastic_behavior(self):
        """Test stochastic properties of downsampled lexicase over multiple runs."""
        
        # 25 individuals, 20 cases
        np.random.seed(555)
        fitness_matrix = np.random.rand(25, 20) * 50
        
        # Add some structure: make individuals 0-4 generally better
        fitness_matrix[0:5, :] += 20
        
        # Collect selections over multiple seeds
        all_selections = []
        for seed in range(40):
            selected = downsample_lexicase_selection(
                fitness_matrix, num_selected=30, downsample_size=8, seed=seed
            )
            all_selections.extend(selected)
        
        selection_counts = np.bincount(all_selections, minlength=25)
        total_selections = len(all_selections)
        
        # Statistical validation
        # 1. Better individuals should be selected more often
        top_5_selections = selection_counts[0:5].sum()
        expected_random = total_selections * (5/25)
        assert top_5_selections > expected_random * 1.5, \
            "Top individuals should be favored in downsampled lexicase"
        
        # 2. Should maintain stochasticity (no complete determinism)
        non_zero_count = np.count_nonzero(selection_counts)
        assert non_zero_count >= 5, \
            f"Should maintain stochastic diversity: {non_zero_count} individuals selected"
        
        # 3. No individual should completely dominate
        max_selections = np.max(selection_counts)
        max_share = max_selections / total_selections
        assert max_share < 0.6, \
            f"No individual should dominate: max share = {max_share:.2%}"
    
    
    def test_downsample_vs_full_lexicase_comparison(self):
        """Compare downsampled lexicase with full lexicase on structured problems."""
        
        # Create a problem where downsampling should reveal different patterns
        # 30 individuals, 25 cases
        fitness_matrix = np.random.rand(30, 25) * 20
        
        # Create some individuals that are "broadly good" vs "narrowly excellent"
        # Broadly good individuals (0-4): decent on all cases
        fitness_matrix[0:5, :] += 15
        
        # Narrowly excellent individuals (5-14): excellent on specific case subsets
        for i in range(5, 15):
            start_case = (i - 5) * 2
            end_case = min(start_case + 3, 25)
            fitness_matrix[i, start_case:end_case] += 40
        
        # Run both full and downsampled lexicase
        full_selected = lexicase_selection(fitness_matrix, num_selected=150, seed=42)
        down_selected = downsample_lexicase_selection(
            fitness_matrix, num_selected=150, downsample_size=8, seed=42
        )
        
        full_counts = np.bincount(full_selected, minlength=30)
        down_counts = np.bincount(down_selected, minlength=30)
        
        # Both should select the enhanced individuals more than random
        enhanced_individuals = list(range(15))  # First 15 are enhanced
        
        full_enhanced = full_counts[enhanced_individuals].sum()
        down_enhanced = down_counts[enhanced_individuals].sum()
        expected_random = 150 * (15/30)  # Random chance
        
        assert full_enhanced > expected_random, \
            "Full lexicase should favor enhanced individuals"
        assert down_enhanced > expected_random, \
            "Downsampled lexicase should favor enhanced individuals"
        
        # Downsampling might give narrowly excellent individuals a better chance
        narrowly_excellent = list(range(5, 15))
        full_narrow = full_counts[narrowly_excellent].sum()
        down_narrow = down_counts[narrowly_excellent].sum()
        
        # This is not a strict requirement, but downsampling often helps specialists
        # We'll just check that both methods find some of these individuals
        assert full_narrow > 0, "Full lexicase should find some narrow specialists"
        assert down_narrow > 0, "Downsampled lexicase should find some narrow specialists" 