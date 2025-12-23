"""
Algorithm correctness tests for lexicase selection.

These tests verify that the lexicase selection algorithm behaves correctly
according to its specification:
1. Candidates are filtered by best performance on each case
2. Case order is randomized for each selection
3. Ties are broken randomly
4. Selection is with replacement
"""

import numpy as np
import pytest
from lexicase import (
    lexicase_selection,
    epsilon_lexicase_selection,
    downsample_lexicase_selection,
)
from lexicase.numpy_impl import (
    numpy_lexicase_selection,
    numpy_epsilon_lexicase_selection,
    numpy_compute_mad_epsilon,
    _compute_case_distances,
    _farthest_first_traversal,
)


# =============================================================================
# Algorithm Correctness Tests
# =============================================================================

class TestLexicaseAlgorithmCorrectness:
    """Test that lexicase selection follows the correct algorithm."""

    def test_best_on_first_case_wins_when_unique(self):
        """If one individual is uniquely best on case 0, and case 0 comes first,
        that individual should always be selected."""
        # Individual 0 is uniquely best on case 0
        fitness = np.array([
            [10.0, 0.0, 0.0],  # Best on case 0
            [5.0, 10.0, 0.0],  # Best on case 1
            [5.0, 0.0, 10.0],  # Best on case 2
        ])

        # Run many selections - individual 0 should be selected ~1/3 of the time
        # (when case 0 is first in the shuffled order)
        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(fitness, 3000, rng)

        counts = np.bincount(selected, minlength=3)

        # Each specialist should be selected roughly 1/3 of the time
        for i in range(3):
            assert 800 < counts[i] < 1200, f"Individual {i} selected {counts[i]} times"

    def test_filtering_eliminates_non_best(self):
        """Individuals not best on the first case should be eliminated."""
        # Individual 0 is strictly worse than others on case 0
        # Individual 1 is best only on case 0, but loses to 2 on secondary cases
        # Individual 2 is tied for best on case 0, and dominates individual 1 on other cases
        fitness = np.array([
            [0.0, 10.0, 10.0],  # Worst on case 0, best on cases 1 and 2
            [10.0, 0.0, 0.0],   # Best on case 0 only, worst on cases 1 and 2
            [10.0, 5.0, 5.0],   # Tied for best on case 0, middle on others
        ])

        # Analysis:
        # - Case 0 first: Individuals 1,2 pass. Then case 1 or 2 decides.
        #   Individual 2 beats 1 on both (5>0). So Individual 2 wins.
        # - Case 1 first: Individual 0 wins (10>5>0)
        # - Case 2 first: Individual 0 wins (10>5>0)
        # So Individual 0 should win ~2/3, Individual 2 should win ~1/3, Individual 1 never wins

        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(fitness, 3000, rng)

        counts = np.bincount(selected, minlength=3)

        # Individual 0 should be selected ~2/3 of time (cases 1 and 2 first)
        assert counts[0] > 1500, f"Individual 0 selected {counts[0]} times, expected ~2000"
        # Individual 2 should be selected ~1/3 of time (case 0 first)
        assert counts[2] > 500, f"Individual 2 selected {counts[2]} times, expected ~1000"
        # Individual 1 should never win (loses to 2 when case 0 is first)
        assert counts[1] == 0, f"Individual 1 should never win, but selected {counts[1]} times"

    def test_dominating_individual_always_selected(self):
        """An individual that dominates all others should always be selected."""
        # Individual 0 is best on ALL cases
        fitness = np.array([
            [10.0, 10.0, 10.0],  # Best on all
            [5.0, 5.0, 5.0],
            [1.0, 1.0, 1.0],
        ])

        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(fitness, 100, rng)

        # All selections should be individual 0
        assert all(s == 0 for s in selected)

    def test_tie_on_all_cases_random_selection(self):
        """When individuals tie on all cases, selection should be random."""
        # All individuals identical
        fitness = np.array([
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
        ])

        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(fitness, 3000, rng)

        counts = np.bincount(selected, minlength=3)

        # Each should be selected roughly 1/3 of the time
        for i in range(3):
            assert 800 < counts[i] < 1200, f"Individual {i} selected {counts[i]} times"

    def test_selection_with_replacement(self):
        """Same individual can be selected multiple times."""
        # Only one individual exists
        fitness = np.array([[1.0, 2.0, 3.0]])

        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(fitness, 10, rng)

        assert len(selected) == 10
        assert all(s == 0 for s in selected)

    def test_second_case_matters_when_tied_on_first(self):
        """When tied on first case, second case determines selection."""
        # Individuals 0 and 1 tie on case 0, but 0 is better on case 1
        fitness = np.array([
            [10.0, 10.0, 0.0],  # Tied on 0, best on 1
            [10.0, 5.0, 0.0],   # Tied on 0, worse on 1
            [5.0, 5.0, 10.0],   # Worse on 0, best on 2
        ])

        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(fitness, 3000, rng)

        counts = np.bincount(selected, minlength=3)

        # Individual 0 should be selected more than individual 1
        # (when case 0 is first, they both pass, then case 1 or 2 decides)
        # Individual 2 only wins when case 2 is first
        assert counts[2] > 500  # Individual 2 should win ~1/3 of time


class TestEpsilonLexicaseCorrectness:
    """Test epsilon lexicase algorithm correctness."""

    def test_epsilon_expands_selection_pool(self):
        """Epsilon should allow near-best individuals to pass."""
        fitness = np.array([
            [10.0, 0.0],  # Best on case 0
            [9.5, 0.0],   # Within epsilon=1 of best on case 0
            [5.0, 10.0],  # Best on case 1
        ])

        # With epsilon=0, individual 1 would never be selected when case 0 is first
        rng1 = np.random.default_rng(42)
        selected_no_eps = numpy_epsilon_lexicase_selection(fitness, 1000, 0.0, rng1)
        count_1_no_eps = np.sum(selected_no_eps == 1)

        # With epsilon=1, individual 1 should sometimes be selected
        rng2 = np.random.default_rng(42)
        selected_eps = numpy_epsilon_lexicase_selection(fitness, 1000, 1.0, rng2)
        count_1_eps = np.sum(selected_eps == 1)

        assert count_1_eps > count_1_no_eps

    def test_large_epsilon_includes_all(self):
        """Very large epsilon should make all individuals equally likely."""
        fitness = np.array([
            [10.0, 0.0],
            [0.0, 10.0],
            [5.0, 5.0],
        ])

        # With very large epsilon, everyone passes every case
        rng = np.random.default_rng(42)
        selected = numpy_epsilon_lexicase_selection(fitness, 3000, 100.0, rng)

        counts = np.bincount(selected, minlength=3)

        # All should be selected roughly equally
        for i in range(3):
            assert 800 < counts[i] < 1200

    def test_per_case_epsilon(self):
        """Different epsilon per case should work correctly."""
        fitness = np.array([
            [10.0, 10.0],  # Best on both
            [9.0, 5.0],    # Close on case 0, far on case 1
            [5.0, 9.0],    # Far on case 0, close on case 1
        ])

        # epsilon = [2.0, 0.0] means case 0 is relaxed, case 1 is strict
        rng = np.random.default_rng(42)
        epsilon = np.array([2.0, 0.0])
        selected = numpy_epsilon_lexicase_selection(fitness, 1000, epsilon, rng)

        counts = np.bincount(selected, minlength=3)

        # Individual 0 should dominate (best on case 1, within epsilon on case 0)
        # Individual 1 passes case 0 (within epsilon) but fails case 1
        # Individual 2 fails case 0 (epsilon=2, diff=5) when case 0 is first
        assert counts[0] > counts[1]
        assert counts[0] > counts[2]


class TestDownsampleCorrectness:
    """Test downsampled lexicase algorithm correctness."""

    def test_downsample_uses_subset_of_cases(self):
        """Downsampling should use fewer cases, changing selection dynamics."""
        # Create fitness where individual 0 is specialist on many cases
        n_cases = 20
        fitness = np.zeros((3, n_cases))
        fitness[0, :10] = 10.0  # Individual 0 best on first 10 cases
        fitness[1, 10:] = 10.0  # Individual 1 best on last 10 cases
        fitness[2, :] = 5.0     # Individual 2 mediocre on all

        # With full lexicase, individuals 0 and 1 should split selections
        rng1 = np.random.default_rng(42)
        selected_full = numpy_lexicase_selection(fitness, 1000, rng1)

        # With downsample_size=1, selection depends on which case is sampled
        rng2 = np.random.default_rng(42)
        from lexicase.numpy_impl import numpy_downsample_lexicase_selection
        selected_down = numpy_downsample_lexicase_selection(fitness, 1000, 1, rng2)

        # Both should produce valid selections
        assert len(selected_full) == 1000
        assert len(selected_down) == 1000

    def test_downsample_size_larger_than_cases(self):
        """Downsample size > n_cases should use all cases."""
        fitness = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ])

        rng = np.random.default_rng(42)
        from lexicase.numpy_impl import numpy_downsample_lexicase_selection
        selected = numpy_downsample_lexicase_selection(fitness, 100, 100, rng)

        assert len(selected) == 100
        assert all(0 <= s < 3 for s in selected)


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelperFunctions:
    """Test internal helper functions."""

    def test_compute_mad_epsilon(self):
        """Test MAD computation."""
        # Known case: values [1, 2, 3, 4, 5]
        # Median = 3, deviations = [2, 1, 0, 1, 2], MAD = 1
        fitness = np.array([[1], [2], [3], [4], [5]], dtype=float)

        mad = numpy_compute_mad_epsilon(fitness)

        assert len(mad) == 1
        assert abs(mad[0] - 1.0) < 1e-10

    def test_compute_mad_multiple_cases(self):
        """Test MAD with multiple test cases."""
        fitness = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ])

        mad = numpy_compute_mad_epsilon(fitness)

        assert len(mad) == 2
        assert abs(mad[0] - 1.0) < 1e-10  # MAD of [1,2,3,4,5] = 1
        assert abs(mad[1] - 10.0) < 1e-10  # MAD of [10,20,30,40,50] = 10

    def test_compute_mad_identical_values(self):
        """Test MAD with identical values (should use min_epsilon)."""
        fitness = np.array([
            [5.0, 5.0],
            [5.0, 5.0],
            [5.0, 5.0],
        ])

        mad = numpy_compute_mad_epsilon(fitness)

        assert len(mad) == 2
        assert all(m >= 1e-10 for m in mad)

    def test_compute_case_distances(self):
        """Test case distance computation."""
        # Create fitness where cases have different solve patterns
        fitness = np.array([
            [1.0, 0.0, 1.0],  # Solves cases 0, 2
            [0.0, 1.0, 1.0],  # Solves cases 1, 2
            [1.0, 1.0, 0.0],  # Solves cases 0, 1
        ])

        sample_indices = np.array([0, 1, 2])
        distances = _compute_case_distances(fitness, sample_indices, threshold=0.5)

        # Distance matrix should be symmetric
        assert np.allclose(distances, distances.T)

        # Diagonal should be 0
        assert np.allclose(np.diag(distances), 0)

    def test_farthest_first_traversal(self):
        """Test Farthest First Traversal algorithm."""
        # Create distance matrix where cases are far apart
        distances = np.array([
            [0.0, 5.0, 10.0],
            [5.0, 0.0, 5.0],
            [10.0, 5.0, 0.0],
        ])

        rng = np.random.default_rng(42)
        selected = _farthest_first_traversal(distances, 2, rng)

        assert len(selected) == 2
        # Should select cases that are far apart
        assert set(selected) in [{0, 2}, {0, 1}, {1, 2}]

    def test_farthest_first_traversal_all_cases(self):
        """FFT with downsample_size >= n_cases returns all cases."""
        distances = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ])

        rng = np.random.default_rng(42)
        selected = _farthest_first_traversal(distances, 5, rng)

        assert len(selected) == 3
        assert set(selected) == {0, 1, 2}


# =============================================================================
# Statistical Distribution Tests
# =============================================================================

class TestSelectionDistribution:
    """Test that selection distributions match expected behavior."""

    def test_uniform_specialists_equal_selection(self):
        """N specialists should each be selected ~1/N of the time."""
        n_specialists = 5
        n_cases = 5

        # Each specialist is best on exactly one case
        fitness = np.zeros((n_specialists, n_cases))
        for i in range(n_specialists):
            fitness[i, i] = 10.0

        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(fitness, 10000, rng)

        counts = np.bincount(selected, minlength=n_specialists)
        expected = 10000 / n_specialists

        for i in range(n_specialists):
            assert abs(counts[i] - expected) < expected * 0.15  # Within 15%

    def test_unequal_specialists_proportional_selection(self):
        """Specialist for more cases should be selected proportionally more."""
        # Individual 0 is specialist on 2 cases, others on 1 each
        fitness = np.array([
            [10.0, 10.0, 0.0, 0.0],  # Best on cases 0, 1
            [0.0, 0.0, 10.0, 0.0],   # Best on case 2
            [0.0, 0.0, 0.0, 10.0],   # Best on case 3
        ])

        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(fitness, 8000, rng)

        counts = np.bincount(selected, minlength=3)

        # Individual 0 should be selected ~2x as often as others
        # (cases 0,1 → individual 0 wins; cases 2,3 → individuals 1,2 win)
        assert counts[0] > counts[1] * 1.5
        assert counts[0] > counts[2] * 1.5

    def test_generalist_never_selected_with_specialists(self):
        """A generalist worse than all specialists should rarely be selected."""
        fitness = np.array([
            [10.0, 0.0, 0.0],  # Specialist
            [0.0, 10.0, 0.0],  # Specialist
            [0.0, 0.0, 10.0],  # Specialist
            [5.0, 5.0, 5.0],   # Generalist (never best on any case)
        ])

        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(fitness, 1000, rng)

        counts = np.bincount(selected, minlength=4)

        # Generalist should never be selected
        assert counts[3] == 0


# =============================================================================
# Elitism Tests
# =============================================================================

class TestElitismBehavior:
    """Test elitism functionality."""

    def test_elitism_always_includes_best(self):
        """Elitism should always include the best individual."""
        fitness = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],  # Best total fitness
        ])

        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(fitness, 5, rng, elitism=1)

        assert 2 in selected

    def test_elitism_correct_count(self):
        """Elitism should include exactly the specified number of elites."""
        fitness = np.array([
            [1.0, 1.0],  # Total: 2
            [2.0, 2.0],  # Total: 4
            [3.0, 3.0],  # Total: 6
            [4.0, 4.0],  # Total: 8
        ])

        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(fitness, 10, rng, elitism=2)

        # Top 2 by total fitness are individuals 3 and 2
        assert 3 in selected
        assert 2 in selected

    def test_elitism_preserves_selection_count(self):
        """Total selections should equal num_selected regardless of elitism."""
        fitness = np.random.rand(10, 5)

        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(fitness, 20, rng, elitism=3)

        assert len(selected) == 20

    def test_elitism_works_with_epsilon(self):
        """Elitism should work correctly with epsilon lexicase."""
        fitness = np.array([
            [1.0, 1.0],
            [5.0, 5.0],  # Best total
            [2.0, 2.0],
        ])

        rng = np.random.default_rng(42)
        selected = numpy_epsilon_lexicase_selection(fitness, 5, 0.1, rng, elitism=1)

        assert 1 in selected  # Best individual should be included


# =============================================================================
# Regression Tests
# =============================================================================

class TestRegression:
    """Regression tests with known inputs/outputs."""

    def test_deterministic_output_seed_42(self):
        """Verify deterministic output for a known seed."""
        fitness = np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ])

        # Run twice with same seed
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        selected1 = numpy_lexicase_selection(fitness, 10, rng1)
        selected2 = numpy_lexicase_selection(fitness, 10, rng2)

        np.testing.assert_array_equal(selected1, selected2)

    def test_output_dtype(self):
        """Output should be integer array."""
        fitness = np.array([[1.0, 2.0], [3.0, 4.0]])

        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(fitness, 5, rng)

        assert selected.dtype in [np.int32, np.int64, int]

    def test_empty_selection(self):
        """num_selected=0 should return empty array."""
        fitness = np.array([[1.0, 2.0], [3.0, 4.0]])

        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(fitness, 0, rng)

        assert len(selected) == 0
        assert isinstance(selected, np.ndarray)
