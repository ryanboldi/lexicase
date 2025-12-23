"""
Extended edge case tests for lexicase selection.

Tests boundary conditions, unusual inputs, and numerical edge cases.
"""

import numpy as np
import pytest
from lexicase import (
    lexicase_selection,
    epsilon_lexicase_selection,
    downsample_lexicase_selection,
    informed_downsample_lexicase_selection,
)
from lexicase.numpy_impl import (
    numpy_lexicase_selection,
    numpy_epsilon_lexicase_selection,
    numpy_downsample_lexicase_selection,
    numpy_compute_mad_epsilon,
)


# =============================================================================
# Numerical Edge Cases
# =============================================================================

class TestNumericalEdgeCases:
    """Test handling of numerical edge cases."""

    def test_very_small_fitness_values(self):
        """Should handle very small fitness values."""
        fitness = np.array([
            [1e-100, 1e-101, 1e-99],
            [1e-101, 1e-100, 1e-101],
            [1e-99, 1e-101, 1e-100],
        ])

        selected = lexicase_selection(fitness, 10, seed=42)

        assert len(selected) == 10
        assert all(0 <= s < 3 for s in selected)

    def test_very_large_fitness_values(self):
        """Should handle very large fitness values."""
        fitness = np.array([
            [1e100, 1e99, 1e98],
            [1e99, 1e100, 1e98],
            [1e98, 1e99, 1e100],
        ])

        selected = lexicase_selection(fitness, 10, seed=42)

        assert len(selected) == 10
        assert all(0 <= s < 3 for s in selected)

    def test_mixed_magnitude_fitness(self):
        """Should handle fitness values of vastly different magnitudes."""
        fitness = np.array([
            [1e10, 1e-10, 1.0],
            [1e-10, 1e10, 1.0],
            [1.0, 1.0, 1e10],
        ])

        selected = lexicase_selection(fitness, 30, seed=42)

        assert len(selected) == 30

    def test_zero_fitness_values(self):
        """Should handle zero fitness values."""
        fitness = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        selected = lexicase_selection(fitness, 30, seed=42)

        assert len(selected) == 30

    def test_all_zero_fitness(self):
        """Should handle all-zero fitness matrix."""
        fitness = np.zeros((5, 5))

        selected = lexicase_selection(fitness, 20, seed=42)

        assert len(selected) == 20
        # All equally bad, should select randomly
        counts = np.bincount(selected, minlength=5)
        assert all(c > 0 for c in counts)

    def test_negative_fitness_values(self):
        """Should handle negative fitness values correctly."""
        fitness = np.array([
            [-1.0, -2.0, -3.0],
            [-2.0, -1.0, -3.0],
            [-3.0, -3.0, -1.0],
        ])

        # Higher is better, so -1 beats -2 beats -3
        selected = lexicase_selection(fitness, 30, seed=42)

        assert len(selected) == 30

    def test_inf_fitness_values(self):
        """Should handle infinity fitness values."""
        fitness = np.array([
            [np.inf, 0.0, 0.0],
            [0.0, np.inf, 0.0],
            [0.0, 0.0, np.inf],
        ])

        selected = lexicase_selection(fitness, 30, seed=42)

        assert len(selected) == 30

    def test_negative_inf_fitness(self):
        """Should handle negative infinity."""
        fitness = np.array([
            [0.0, -np.inf, 0.0],
            [-np.inf, 0.0, 0.0],
            [0.0, 0.0, -np.inf],
        ])

        # Each individual is worst on exactly one case
        selected = lexicase_selection(fitness, 30, seed=42)

        assert len(selected) == 30

    def test_epsilon_with_small_values(self):
        """Epsilon should work with very small fitness differences."""
        fitness = np.array([
            [1.0, 1.0],
            [1.0 + 1e-10, 1.0],  # Slightly better on case 0
            [1.0, 1.0 + 1e-10],  # Slightly better on case 1
        ])

        # With epsilon=0, tiny differences matter
        rng = np.random.default_rng(42)
        selected_no_eps = numpy_epsilon_lexicase_selection(fitness, 1000, 0.0, rng)

        # With small epsilon, they should be treated as equal
        rng = np.random.default_rng(42)
        selected_eps = numpy_epsilon_lexicase_selection(fitness, 1000, 1e-9, rng)

        # Both should produce valid results
        assert len(selected_no_eps) == 1000
        assert len(selected_eps) == 1000


# =============================================================================
# Shape Edge Cases
# =============================================================================

class TestShapeEdgeCases:
    """Test unusual matrix shapes."""

    def test_single_individual_single_case(self):
        """Should handle 1x1 fitness matrix."""
        fitness = np.array([[5.0]])

        selected = lexicase_selection(fitness, 10, seed=42)

        assert len(selected) == 10
        assert all(s == 0 for s in selected)

    def test_many_individuals_single_case(self):
        """Should handle many individuals with one case."""
        n = 100
        fitness = np.arange(n).reshape(n, 1).astype(float)

        selected = lexicase_selection(fitness, 50, seed=42)

        assert len(selected) == 50
        # Individual n-1 is best and should always be selected
        assert all(s == n - 1 for s in selected)

    def test_single_individual_many_cases(self):
        """Should handle one individual with many cases."""
        fitness = np.random.rand(1, 100)

        selected = lexicase_selection(fitness, 10, seed=42)

        assert len(selected) == 10
        assert all(s == 0 for s in selected)

    def test_wide_matrix(self):
        """Should handle matrix with many more cases than individuals."""
        fitness = np.random.rand(5, 1000)

        selected = lexicase_selection(fitness, 20, seed=42)

        assert len(selected) == 20

    def test_tall_matrix(self):
        """Should handle matrix with many more individuals than cases."""
        fitness = np.random.rand(1000, 5)

        selected = lexicase_selection(fitness, 50, seed=42)

        assert len(selected) == 50
        assert all(0 <= s < 1000 for s in selected)

    def test_square_matrix(self):
        """Should handle square matrix."""
        n = 50
        fitness = np.random.rand(n, n)

        selected = lexicase_selection(fitness, 100, seed=42)

        assert len(selected) == 100


# =============================================================================
# Data Type Edge Cases
# =============================================================================

class TestDataTypeEdgeCases:
    """Test various input data types."""

    def test_float32_input(self):
        """Should work with float32 input."""
        fitness = np.random.rand(10, 5).astype(np.float32)

        selected = lexicase_selection(fitness, 20, seed=42)

        assert len(selected) == 20

    def test_float64_input(self):
        """Should work with float64 input."""
        fitness = np.random.rand(10, 5).astype(np.float64)

        selected = lexicase_selection(fitness, 20, seed=42)

        assert len(selected) == 20

    def test_int32_input(self):
        """Should work with int32 input."""
        fitness = np.random.randint(0, 100, size=(10, 5)).astype(np.int32)

        selected = lexicase_selection(fitness, 20, seed=42)

        assert len(selected) == 20

    def test_int64_input(self):
        """Should work with int64 input."""
        fitness = np.random.randint(0, 100, size=(10, 5)).astype(np.int64)

        selected = lexicase_selection(fitness, 20, seed=42)

        assert len(selected) == 20

    def test_list_input(self):
        """Should work with nested list input."""
        fitness = [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ]

        selected = lexicase_selection(fitness, 10, seed=42)

        assert len(selected) == 10

    def test_tuple_input(self):
        """Should work with tuple input."""
        fitness = (
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.5, 0.5),
        )

        selected = lexicase_selection(fitness, 10, seed=42)

        assert len(selected) == 10


# =============================================================================
# Selection Count Edge Cases
# =============================================================================

class TestSelectionCountEdgeCases:
    """Test edge cases for num_selected parameter."""

    def test_select_exactly_population_size(self):
        """Should work when selecting exactly population size."""
        fitness = np.random.rand(10, 5)

        selected = lexicase_selection(fitness, 10, seed=42)

        assert len(selected) == 10

    def test_select_much_more_than_population(self):
        """Should work when selecting many more than population."""
        fitness = np.random.rand(3, 3)

        selected = lexicase_selection(fitness, 1000, seed=42)

        assert len(selected) == 1000

    def test_select_one(self):
        """Should work when selecting exactly one."""
        fitness = np.random.rand(10, 5)

        selected = lexicase_selection(fitness, 1, seed=42)

        assert len(selected) == 1


# =============================================================================
# Elitism Edge Cases
# =============================================================================

class TestElitismEdgeCases:
    """Test elitism edge cases."""

    def test_elitism_equals_num_selected(self):
        """Should work when elitism equals num_selected."""
        fitness = np.random.rand(10, 5)

        selected = lexicase_selection(fitness, 5, seed=42, elitism=5)

        assert len(selected) == 5
        # All should be elite
        total_fitness = np.sum(fitness, axis=1)
        top5 = set(np.argsort(total_fitness)[-5:])
        assert set(selected) == top5

    def test_elitism_with_tied_fitness(self):
        """Should handle ties when selecting elites."""
        # All have same total fitness
        fitness = np.array([
            [2.0, 2.0],
            [1.0, 3.0],
            [3.0, 1.0],
            [2.0, 2.0],
        ])

        selected = lexicase_selection(fitness, 5, seed=42, elitism=2)

        assert len(selected) == 5


# =============================================================================
# Downsample Edge Cases
# =============================================================================

class TestDownsampleEdgeCases:
    """Test downsampling edge cases."""

    def test_downsample_size_equals_cases(self):
        """Should work when downsample_size equals n_cases."""
        fitness = np.random.rand(10, 5)

        selected = downsample_lexicase_selection(fitness, 20, 5, seed=42)

        assert len(selected) == 20

    def test_downsample_size_one(self):
        """Should work with downsample_size=1."""
        fitness = np.random.rand(10, 20)

        selected = downsample_lexicase_selection(fitness, 50, 1, seed=42)

        assert len(selected) == 50


# =============================================================================
# Informed Downsample Edge Cases
# =============================================================================

class TestInformedDownsampleEdgeCases:
    """Test informed downsampling edge cases."""

    def test_sample_rate_very_small(self):
        """Should work with very small sample rate."""
        fitness = np.random.rand(100, 20)

        # 0.01 rate on 100 individuals = 1 sample
        selected = informed_downsample_lexicase_selection(
            fitness, 20, 5, seed=42, sample_rate=0.01
        )

        assert len(selected) == 20

    def test_sample_rate_one(self):
        """Should work with sample_rate=1.0."""
        fitness = np.random.rand(50, 10)

        # Sample all individuals
        selected = informed_downsample_lexicase_selection(
            fitness, 20, 5, seed=42, sample_rate=1.0
        )

        assert len(selected) == 20

    def test_threshold_zero(self):
        """Should work with threshold=0."""
        fitness = np.random.rand(20, 10)

        selected = informed_downsample_lexicase_selection(
            fitness, 10, 3, seed=42, threshold=0.0
        )

        assert len(selected) == 10

    def test_threshold_very_high(self):
        """Should work with very high threshold."""
        fitness = np.random.rand(20, 10)

        # No one passes the threshold
        selected = informed_downsample_lexicase_selection(
            fitness, 10, 3, seed=42, threshold=100.0
        )

        assert len(selected) == 10


# =============================================================================
# MAD Edge Cases
# =============================================================================

class TestMADEdgeCases:
    """Test MAD computation edge cases."""

    def test_mad_two_values(self):
        """MAD with two values."""
        fitness = np.array([[1.0], [3.0]])

        mad = numpy_compute_mad_epsilon(fitness)

        # Median = 2, deviations = [1, 1], MAD = 1
        assert abs(mad[0] - 1.0) < 1e-10

    def test_mad_single_value(self):
        """MAD with single value should be min_epsilon."""
        fitness = np.array([[5.0]])

        mad = numpy_compute_mad_epsilon(fitness)

        assert mad[0] >= 1e-10

    def test_mad_with_outliers(self):
        """MAD should be robust to outliers."""
        # 99 values of 1.0, one outlier of 1000.0
        values = np.ones((100, 1))
        values[0, 0] = 1000.0

        mad = numpy_compute_mad_epsilon(values)

        # MAD should be close to 0 (most values identical)
        # but at least min_epsilon
        assert mad[0] >= 1e-10


# =============================================================================
# Stress Tests
# =============================================================================

class TestStress:
    """Stress tests with large data."""

    def test_large_population(self):
        """Should handle large population efficiently."""
        np.random.seed(42)
        fitness = np.random.rand(10000, 20)

        selected = lexicase_selection(fitness, 100, seed=42)

        assert len(selected) == 100
        assert all(0 <= s < 10000 for s in selected)

    def test_many_cases(self):
        """Should handle many test cases."""
        np.random.seed(42)
        fitness = np.random.rand(100, 1000)

        selected = lexicase_selection(fitness, 50, seed=42)

        assert len(selected) == 50

    def test_many_selections(self):
        """Should handle many selections efficiently."""
        np.random.seed(42)
        fitness = np.random.rand(50, 20)

        selected = lexicase_selection(fitness, 10000, seed=42)

        assert len(selected) == 10000

    def test_all_large(self):
        """Stress test with large population, cases, and selections."""
        np.random.seed(42)
        fitness = np.random.rand(1000, 100)

        selected = lexicase_selection(fitness, 500, seed=42)

        assert len(selected) == 500
