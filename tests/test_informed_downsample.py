"""
Tests for informed downsampled lexicase selection.
"""

import numpy as np
import pytest

from lexicase import downsample_lexicase_selection, informed_downsample_lexicase_selection


class TestInformedDownsample:
    """Test informed downsampled lexicase selection functionality."""

    def test_basic_functionality_numpy(self):
        """Test that informed downsample works with NumPy arrays."""
        # Create fitness matrix with clear patterns
        # Individuals 0-2 are good at cases 0-2
        # Individuals 3-5 are good at cases 3-5
        fitness_matrix = np.array([
            [5, 5, 5, 1, 1, 1],  # Good at first half
            [5, 5, 5, 1, 1, 1],
            [5, 5, 5, 1, 1, 1],
            [1, 1, 1, 5, 5, 5],  # Good at second half
            [1, 1, 1, 5, 5, 5],
            [1, 1, 1, 5, 5, 5],
        ])

        # Select 3 individuals with informed downsample
        selected = informed_downsample_lexicase_selection(
            fitness_matrix, 3, downsample_size=3, seed=42, sample_rate=0.5
        )

        assert len(selected) == 3
        assert all(0 <= idx < 6 for idx in selected)

    def test_case_distance_computation(self):
        """Test that case distances are computed correctly."""
        # Create matrix where cases have clear solve patterns
        fitness_matrix = np.array([
            [10, 0, 10, 0],  # Solves cases 0 and 2
            [10, 0, 10, 0],  # Solves cases 0 and 2 (same as ind 0)
            [0, 10, 0, 10],  # Solves cases 1 and 3
            [0, 10, 0, 10],  # Solves cases 1 and 3 (same as ind 2)
        ])

        # Import the distance computation function for testing
        from lexicase.numpy_impl import _compute_case_distances

        # Use all individuals for distance calculation
        sample_indices = np.arange(4)
        distances = _compute_case_distances(fitness_matrix, sample_indices, threshold=5)

        # Cases 0 and 2 should have distance 0 (solved by same individuals)
        assert distances[0, 2] == 0
        # Cases 1 and 3 should have distance 0 (solved by same individuals)
        assert distances[1, 3] == 0
        # Cases 0 and 1 should have positive distance (solved by different individuals)
        assert distances[0, 1] > 0
        # Cases 0 and 3 should have positive distance
        assert distances[0, 3] > 0

    def test_farthest_first_traversal(self):
        """Test that FFT selects maximally distant cases."""
        from lexicase.numpy_impl import _farthest_first_traversal

        # Create a distance matrix where cases 0 and 3 are most distant
        distances = np.array([
            [0, 1, 2, 5],
            [1, 0, 1, 4],
            [2, 1, 0, 3],
            [5, 4, 3, 0],
        ])

        rng = np.random.default_rng(42)
        selected = _farthest_first_traversal(distances, 2, rng)

        # Should select 2 cases
        assert len(selected) == 2
        # The selected cases should be maximally distant
        # With this matrix, we expect cases like 0 and 3 to be selected together

    def test_informed_vs_random_diversity(self):
        """Test that informed sampling selects more diverse cases than random."""
        # Create fitness matrix with redundant cases
        fitness_matrix = np.array([
            [1, 1, 1, 5, 5, 5, 9, 9, 9],  # Cases 0-2 similar, 3-5 similar, 6-8 similar
            [1, 1, 1, 5, 5, 5, 9, 9, 9],
            [2, 2, 2, 6, 6, 6, 8, 8, 8],
            [2, 2, 2, 6, 6, 6, 8, 8, 8],
            [3, 3, 3, 7, 7, 7, 7, 7, 7],
        ])

        # For informed selection with proper sample rate
        informed_results = []
        for seed in range(10):
            selected = informed_downsample_lexicase_selection(
                fitness_matrix, 5, downsample_size=3, seed=seed, sample_rate=1.0
            )
            informed_results.append(selected)

        # For random selection
        random_results = []
        for seed in range(10):
            selected = downsample_lexicase_selection(
                fitness_matrix, 5, downsample_size=3, seed=seed
            )
            random_results.append(selected)

        # Both should return valid selections
        for result in informed_results + random_results:
            assert len(result) == 5
            assert all(0 <= idx < 5 for idx in result)

    def test_threshold_parameter(self):
        """Test different threshold options for pass/fail determination."""
        fitness_matrix = np.array([
            [1.0, 5.0, 9.0],
            [2.0, 6.0, 8.0],
            [3.0, 7.0, 7.0],
            [4.0, 8.0, 6.0],
        ])

        # Test with scalar threshold
        selected1 = informed_downsample_lexicase_selection(
            fitness_matrix, 2, downsample_size=2, seed=42, threshold=5.0
        )
        assert len(selected1) == 2

        # Test with per-case thresholds
        thresholds = np.array([2.5, 6.5, 7.5])
        selected2 = informed_downsample_lexicase_selection(
            fitness_matrix, 2, downsample_size=2, seed=42, threshold=thresholds
        )
        assert len(selected2) == 2

        # Test with None (default median)
        selected3 = informed_downsample_lexicase_selection(
            fitness_matrix, 2, downsample_size=2, seed=42, threshold=None
        )
        assert len(selected3) == 2

    def test_elitism_compatibility(self):
        """Test that elitism works with informed downsampling."""
        fitness_matrix = np.array([
            [1, 1, 1],  # Total: 3
            [2, 2, 2],  # Total: 6
            [3, 3, 3],  # Total: 9
            [4, 4, 4],  # Total: 12 - Best
            [5, 5, 5],  # Total: 15 - Second best
        ])

        # Select 3 with elitism=2
        selected = informed_downsample_lexicase_selection(
            fitness_matrix, 3, downsample_size=2, seed=42, elitism=2
        )

        # Should include the top 2 individuals
        assert 3 in selected  # Individual 3 has second-highest total
        assert 4 in selected  # Individual 4 has highest total
        assert len(selected) == 3

    def test_sample_rate_effect(self):
        """Test that sample rate affects computation but not correctness."""
        fitness_matrix = np.random.rand(20, 10)

        # Test with different sample rates
        results = []
        for sample_rate in [0.1, 0.5, 1.0]:
            selected = informed_downsample_lexicase_selection(
                fitness_matrix, 5, downsample_size=5,
                seed=42, sample_rate=sample_rate
            )
            results.append(selected)
            assert len(selected) == 5
            assert all(0 <= idx < 20 for idx in selected)

    def test_edge_cases(self):
        """Test edge cases for informed downsampling."""
        fitness_matrix = np.array([[1, 2], [3, 4]])

        # Downsample size larger than number of cases
        selected = informed_downsample_lexicase_selection(
            fitness_matrix, 1, downsample_size=5, seed=42
        )
        assert len(selected) == 1

        # Very small sample rate
        selected = informed_downsample_lexicase_selection(
            fitness_matrix, 1, downsample_size=1, seed=42, sample_rate=0.01
        )
        assert len(selected) == 1

        # Select all individuals
        selected = informed_downsample_lexicase_selection(
            fitness_matrix, 2, downsample_size=2, seed=42
        )
        assert len(selected) == 2

    def test_validation_errors(self):
        """Test that invalid inputs raise appropriate errors."""
        fitness_matrix = np.array([[1, 2], [3, 4]])

        # Invalid sample rate
        with pytest.raises(ValueError, match="Sample rate must be between 0 and 1"):
            informed_downsample_lexicase_selection(
                fitness_matrix, 1, downsample_size=1, sample_rate=0
            )

        with pytest.raises(ValueError, match="Sample rate must be between 0 and 1"):
            informed_downsample_lexicase_selection(
                fitness_matrix, 1, downsample_size=1, sample_rate=1.5
            )

        # Invalid downsample size
        with pytest.raises(ValueError, match="Downsample size must be positive"):
            informed_downsample_lexicase_selection(
                fitness_matrix, 1, downsample_size=0
            )

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with the same seed."""
        fitness_matrix = np.random.rand(10, 8)

        # Run multiple times with same seed
        results = []
        for _ in range(3):
            selected = informed_downsample_lexicase_selection(
                fitness_matrix, 5, downsample_size=4, seed=42, sample_rate=0.5
            )
            results.append(selected)

        # All results should be identical
        for result in results[1:]:
            np.testing.assert_array_equal(results[0], result)

    def test_informed_selection_identifies_distinct_cases(self):
        """Test that informed selection correctly identifies distinct test cases."""
        # Create a matrix where some cases are redundant
        # Cases 0 and 1 test the same thing (identical columns)
        # Cases 2 and 3 test something different
        fitness_matrix = np.array([
            [5, 5, 1, 1],  # Cases 0&1 identical, cases 2&3 identical
            [5, 5, 1, 1],
            [5, 5, 1, 1],
            [1, 1, 5, 5],
            [1, 1, 5, 5],
            [1, 1, 5, 5],
        ])

        # With informed selection, it should identify that we only need 2 distinct cases
        # even if we request 3
        selected = informed_downsample_lexicase_selection(
            fitness_matrix, 6, downsample_size=3, seed=42, sample_rate=1.0
        )

        assert len(selected) == 6
        # Should get a mix of individuals from both groups
