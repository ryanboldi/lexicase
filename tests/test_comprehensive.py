"""
Comprehensive tests for lexicase selection library.

Tests cover:
1. All selection algorithms (lexicase, epsilon, downsample, informed downsample)
2. Edge cases and error handling
3. Determinism with seeds
4. Elitism functionality
"""

import numpy as np
import pytest

# Import the main API
import lexicase
from lexicase import (
    downsample_lexicase_selection,
    epsilon_lexicase_selection,
    informed_downsample_lexicase_selection,
    lexicase_selection,
)

# Import NumPy implementations directly
from lexicase.numpy_impl import (
    numpy_compute_mad_epsilon,
    numpy_downsample_lexicase_selection,
    numpy_epsilon_lexicase_selection,
    numpy_epsilon_lexicase_selection_with_mad,
    numpy_informed_downsample_lexicase_selection,
    numpy_lexicase_selection,
)

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_fitness():
    """Simple 3x3 fitness matrix for basic tests."""
    return np.array([
        [1.0, 0.0, 1.0],  # Individual 0: specialist for cases 0,2
        [0.0, 1.0, 0.0],  # Individual 1: specialist for case 1
        [0.5, 0.5, 0.5],  # Individual 2: generalist
    ])


@pytest.fixture
def specialist_fitness():
    """Fitness matrix with clear specialists."""
    return np.array([
        [10.0, 0.0, 0.0],  # Specialist for case 0 - total = 10
        [0.0, 10.0, 0.0],  # Specialist for case 1 - total = 10
        [0.0, 0.0, 10.0],  # Specialist for case 2 - total = 10
        [5.0, 5.0, 5.0],   # Generalist - total = 15 (highest!)
    ])


@pytest.fixture
def large_fitness():
    """Larger fitness matrix for stress tests."""
    np.random.seed(42)
    return np.random.rand(100, 50)


@pytest.fixture
def identical_fitness():
    """Fitness matrix where all individuals are identical."""
    return np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])


@pytest.fixture
def close_fitness():
    """Fitness matrix with close values for epsilon testing."""
    return np.array([
        [100.0, 1.0, 1.0],
        [99.0, 2.0, 2.0],   # Within epsilon=2 of best on case 0
        [1.0, 100.0, 1.0],
        [2.0, 99.0, 2.0],   # Within epsilon=2 of best on case 1
    ])


# =============================================================================
# Lexicase Selection Tests
# =============================================================================

class TestLexicaseSelection:
    """Tests for lexicase selection."""

    def test_basic_selection(self, simple_fitness):
        """Test basic selection returns valid indices."""
        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(simple_fitness, 5, rng)

        assert len(selected) == 5
        assert all(0 <= idx < 3 for idx in selected)
        assert isinstance(selected, np.ndarray)

    def test_deterministic_with_seed(self, simple_fitness):
        """Test that same seed produces same results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        selected1 = numpy_lexicase_selection(simple_fitness, 10, rng1)
        selected2 = numpy_lexicase_selection(simple_fitness, 10, rng2)

        np.testing.assert_array_equal(selected1, selected2)

    def test_different_seeds_different_results(self, simple_fitness):
        """Test that different seeds produce different results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)

        selected1 = numpy_lexicase_selection(simple_fitness, 20, rng1)
        selected2 = numpy_lexicase_selection(simple_fitness, 20, rng2)

        assert not np.array_equal(selected1, selected2)

    def test_select_zero(self, simple_fitness):
        """Test selecting zero individuals."""
        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(simple_fitness, 0, rng)

        assert len(selected) == 0
        assert isinstance(selected, np.ndarray)

    def test_single_individual(self):
        """Test with single individual population."""
        fitness = np.array([[1.0, 0.5, 0.8]])
        rng = np.random.default_rng(42)

        selected = numpy_lexicase_selection(fitness, 5, rng)

        assert len(selected) == 5
        assert all(idx == 0 for idx in selected)

    def test_single_case(self):
        """Test with single test case."""
        fitness = np.array([[1.0], [0.5], [0.0]])
        rng = np.random.default_rng(42)

        selected = numpy_lexicase_selection(fitness, 1, rng)

        assert len(selected) == 1
        assert selected[0] == 0  # Best individual

    def test_specialists_get_selected(self, specialist_fitness):
        """Test that specialists are selected when their case comes first."""
        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(specialist_fitness, 100, rng)

        unique = set(selected)
        assert len(unique) >= 3  # At least 3 different individuals

    def test_identical_individuals(self, identical_fitness):
        """Test with identical individuals (should select randomly)."""
        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(identical_fitness, 30, rng)

        assert len(selected) == 30
        counts = np.bincount(selected, minlength=3)
        assert all(c > 0 for c in counts)  # Each selected at least once

    def test_elitism_basic(self, specialist_fitness):
        """Test elitism includes best individuals."""
        rng = np.random.default_rng(42)

        selected = numpy_lexicase_selection(specialist_fitness, 5, rng, elitism=1)

        assert len(selected) == 5
        assert 3 in selected  # Generalist should be in selection

    def test_elitism_multiple(self, specialist_fitness):
        """Test elitism with multiple elite individuals."""
        rng = np.random.default_rng(42)

        selected = numpy_lexicase_selection(specialist_fitness, 5, rng, elitism=2)

        assert len(selected) == 5

    def test_large_population(self, large_fitness):
        """Stress test with large population."""
        rng = np.random.default_rng(42)
        selected = numpy_lexicase_selection(large_fitness, 50, rng)

        assert len(selected) == 50
        assert all(0 <= idx < 100 for idx in selected)


# =============================================================================
# Epsilon Lexicase Tests
# =============================================================================

class TestEpsilonLexicase:
    """Tests for epsilon lexicase selection."""

    def test_basic_epsilon(self, simple_fitness):
        """Test basic epsilon selection."""
        rng = np.random.default_rng(42)
        selected = numpy_epsilon_lexicase_selection(simple_fitness, 5, 0.1, rng)

        assert len(selected) == 5
        assert all(0 <= idx < 3 for idx in selected)

    def test_zero_epsilon_like_regular(self, simple_fitness):
        """Test that epsilon=0 behaves like regular lexicase."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        selected_regular = numpy_lexicase_selection(simple_fitness, 10, rng1)
        selected_epsilon = numpy_epsilon_lexicase_selection(simple_fitness, 10, 0.0, rng2)

        np.testing.assert_array_equal(selected_regular, selected_epsilon)

    def test_large_epsilon_more_diversity(self, close_fitness):
        """Test that large epsilon allows more individuals to pass."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        selected_small = numpy_epsilon_lexicase_selection(close_fitness, 50, 0.5, rng1)
        selected_large = numpy_epsilon_lexicase_selection(close_fitness, 50, 5.0, rng2)

        diversity_small = len(set(selected_small))
        diversity_large = len(set(selected_large))

        assert diversity_large >= diversity_small

    def test_per_case_epsilon(self, simple_fitness):
        """Test epsilon as array (different per case)."""
        rng = np.random.default_rng(42)
        epsilon = np.array([0.1, 0.2, 0.3])

        selected = numpy_epsilon_lexicase_selection(simple_fitness, 5, epsilon, rng)

        assert len(selected) == 5

    def test_elitism_with_epsilon(self, specialist_fitness):
        """Test elitism works with epsilon lexicase."""
        rng = np.random.default_rng(42)
        selected = numpy_epsilon_lexicase_selection(specialist_fitness, 5, 0.1, rng, elitism=1)

        assert len(selected) == 5
        assert 3 in selected


# =============================================================================
# MAD Epsilon Tests
# =============================================================================

class TestMADEpsilon:
    """Tests for MAD-based epsilon computation."""

    def test_compute_mad_basic(self):
        """Test basic MAD computation."""
        fitness = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ])

        mad = numpy_compute_mad_epsilon(fitness)

        assert len(mad) == 2
        assert all(m > 0 for m in mad)

    def test_compute_mad_identical(self):
        """Test MAD with identical values returns minimum epsilon."""
        fitness = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ])

        mad = numpy_compute_mad_epsilon(fitness)

        assert len(mad) == 2
        assert all(m >= 1e-10 for m in mad)

    def test_epsilon_none_uses_mad(self, simple_fitness):
        """Test that epsilon=None in API uses MAD."""
        rng = np.random.default_rng(42)
        selected = numpy_epsilon_lexicase_selection_with_mad(simple_fitness, 5, rng)

        assert len(selected) == 5


# =============================================================================
# Downsample Lexicase Tests
# =============================================================================

class TestDownsampleLexicase:
    """Tests for downsampled lexicase selection."""

    def test_basic_downsample(self, simple_fitness):
        """Test basic downsampled selection."""
        rng = np.random.default_rng(42)
        selected = numpy_downsample_lexicase_selection(simple_fitness, 5, 2, rng)

        assert len(selected) == 5
        assert all(0 <= idx < 3 for idx in selected)

    def test_downsample_larger_than_cases(self, simple_fitness):
        """Test downsample size larger than number of cases."""
        rng = np.random.default_rng(42)
        selected = numpy_downsample_lexicase_selection(simple_fitness, 5, 10, rng)

        assert len(selected) == 5

    def test_downsample_single_case(self, specialist_fitness):
        """Test with downsample_size=1."""
        rng = np.random.default_rng(42)
        selected = numpy_downsample_lexicase_selection(specialist_fitness, 20, 1, rng)

        assert len(selected) == 20
        unique = set(selected)
        assert len(unique) >= 2

    def test_downsample_with_elitism(self, specialist_fitness):
        """Test elitism with downsampled lexicase."""
        rng = np.random.default_rng(42)
        selected = numpy_downsample_lexicase_selection(specialist_fitness, 5, 2, rng, elitism=1)

        assert len(selected) == 5
        assert 3 in selected


# =============================================================================
# Informed Downsample Tests
# =============================================================================

class TestInformedDownsample:
    """Tests for informed downsampled lexicase selection."""

    def test_basic_informed(self, specialist_fitness):
        """Test basic informed downsampled selection."""
        rng = np.random.default_rng(42)
        selected = numpy_informed_downsample_lexicase_selection(
            specialist_fitness, 5, 2, rng
        )

        assert len(selected) == 5
        assert all(0 <= idx < 4 for idx in selected)

    def test_informed_with_threshold(self, specialist_fitness):
        """Test informed downsampling with threshold."""
        rng = np.random.default_rng(42)
        selected = numpy_informed_downsample_lexicase_selection(
            specialist_fitness, 5, 2, rng, threshold=5.0
        )

        assert len(selected) == 5

    def test_informed_sample_rate(self, large_fitness):
        """Test different sample rates."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        selected1 = numpy_informed_downsample_lexicase_selection(
            large_fitness, 10, 5, rng1, sample_rate=0.01
        )
        selected2 = numpy_informed_downsample_lexicase_selection(
            large_fitness, 10, 5, rng2, sample_rate=0.5
        )

        assert len(selected1) == 10
        assert len(selected2) == 10

    def test_informed_with_elitism(self, specialist_fitness):
        """Test elitism with informed downsampling."""
        rng = np.random.default_rng(42)
        selected = numpy_informed_downsample_lexicase_selection(
            specialist_fitness, 5, 2, rng, elitism=1
        )

        assert len(selected) == 5
        assert 3 in selected


# =============================================================================
# Dispatch Layer Tests
# =============================================================================

class TestDispatch:
    """Test dispatch layer."""

    def test_lexicase_dispatch(self, simple_fitness):
        """Test lexicase_selection dispatches correctly."""
        selected = lexicase_selection(simple_fitness, 5, seed=42)

        assert isinstance(selected, np.ndarray)
        assert len(selected) == 5

    def test_epsilon_lexicase_dispatch(self, simple_fitness):
        """Test epsilon_lexicase_selection dispatches correctly."""
        selected = epsilon_lexicase_selection(simple_fitness, 5, epsilon=0.1, seed=42)

        assert isinstance(selected, np.ndarray)
        assert len(selected) == 5

    def test_epsilon_none_dispatch(self, simple_fitness):
        """Test epsilon=None uses MAD."""
        selected = epsilon_lexicase_selection(simple_fitness, 5, epsilon=None, seed=42)

        assert isinstance(selected, np.ndarray)
        assert len(selected) == 5

    def test_downsample_dispatch(self, simple_fitness):
        """Test downsample_lexicase_selection dispatches correctly."""
        selected = downsample_lexicase_selection(simple_fitness, 5, 2, seed=42)

        assert isinstance(selected, np.ndarray)
        assert len(selected) == 5

    def test_informed_downsample_dispatch(self, specialist_fitness):
        """Test informed_downsample_lexicase_selection dispatches correctly."""
        selected = informed_downsample_lexicase_selection(
            specialist_fitness, 5, 2, seed=42
        )

        assert isinstance(selected, np.ndarray)
        assert len(selected) == 5


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_num_selected_negative(self, simple_fitness):
        """Test error for negative num_selected."""
        with pytest.raises(ValueError, match="non-negative"):
            lexicase_selection(simple_fitness, -1, seed=42)

    def test_invalid_seed_type(self, simple_fitness):
        """Test error for invalid seed type."""
        with pytest.raises(ValueError, match="integer"):
            lexicase_selection(simple_fitness, 5, seed="invalid")

    def test_invalid_fitness_1d(self):
        """Test error for 1D fitness matrix."""
        fitness_1d = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="2-dimensional"):
            lexicase_selection(fitness_1d, 5, seed=42)

    def test_invalid_fitness_empty_individuals(self):
        """Test error for empty population."""
        fitness_empty = np.array([]).reshape(0, 3)
        with pytest.raises(ValueError, match="at least one individual"):
            lexicase_selection(fitness_empty, 5, seed=42)

    def test_invalid_fitness_empty_cases(self):
        """Test error for no test cases."""
        fitness_empty = np.array([]).reshape(3, 0)
        with pytest.raises(ValueError, match="at least one test case"):
            lexicase_selection(fitness_empty, 5, seed=42)

    def test_invalid_epsilon_negative(self, simple_fitness):
        """Test error for negative epsilon."""
        with pytest.raises(ValueError, match="non-negative"):
            epsilon_lexicase_selection(simple_fitness, 5, epsilon=-0.1, seed=42)

    def test_invalid_epsilon_array_wrong_length(self, simple_fitness):
        """Test error for wrong epsilon array length."""
        with pytest.raises(ValueError, match="length"):
            epsilon_lexicase_selection(simple_fitness, 5, epsilon=[0.1, 0.2], seed=42)

    def test_invalid_epsilon_array_negative(self, simple_fitness):
        """Test error for negative values in epsilon array."""
        with pytest.raises(ValueError, match="non-negative"):
            epsilon_lexicase_selection(simple_fitness, 5, epsilon=[0.1, -0.1, 0.1], seed=42)

    def test_invalid_downsample_zero(self, simple_fitness):
        """Test error for zero downsample size."""
        with pytest.raises(ValueError, match="positive"):
            downsample_lexicase_selection(simple_fitness, 5, 0, seed=42)

    def test_invalid_downsample_negative(self, simple_fitness):
        """Test error for negative downsample size."""
        with pytest.raises(ValueError, match="positive"):
            downsample_lexicase_selection(simple_fitness, 5, -1, seed=42)

    def test_invalid_sample_rate_zero(self, simple_fitness):
        """Test error for zero sample rate."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            informed_downsample_lexicase_selection(simple_fitness, 5, 2, seed=42, sample_rate=0)

    def test_invalid_sample_rate_over_one(self, simple_fitness):
        """Test error for sample rate > 1."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            informed_downsample_lexicase_selection(simple_fitness, 5, 2, seed=42, sample_rate=1.5)

    def test_invalid_elitism_negative(self, simple_fitness):
        """Test error for negative elitism."""
        with pytest.raises(ValueError, match="non-negative"):
            lexicase_selection(simple_fitness, 5, seed=42, elitism=-1)

    def test_invalid_elitism_exceeds_num_selected(self, simple_fitness):
        """Test error for elitism > num_selected."""
        with pytest.raises(ValueError, match="exceed num_selected"):
            lexicase_selection(simple_fitness, 5, seed=42, elitism=6)

    def test_invalid_elitism_exceeds_population(self, simple_fitness):
        """Test error for elitism > population size."""
        with pytest.raises(ValueError, match="exceed number of individuals"):
            lexicase_selection(simple_fitness, 5, seed=42, elitism=4)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_select_more_than_population(self, simple_fitness):
        """Test selecting more individuals than population (with replacement)."""
        selected = lexicase_selection(simple_fitness, 100, seed=42)

        assert len(selected) == 100
        assert all(0 <= idx < 3 for idx in selected)

    def test_extreme_fitness_values(self):
        """Test with extreme fitness values."""
        fitness = np.array([
            [1e10, 0.0, 0.0],
            [0.0, 1e10, 0.0],
            [0.0, 0.0, 1e10],
        ])

        selected = lexicase_selection(fitness, 10, seed=42)

        assert len(selected) == 10
        assert all(0 <= idx < 3 for idx in selected)

    def test_negative_fitness_values(self):
        """Test with negative fitness values."""
        fitness = np.array([
            [-1.0, -5.0, -2.0],
            [-2.0, -1.0, -5.0],
            [-5.0, -2.0, -1.0],
        ])

        selected = lexicase_selection(fitness, 10, seed=42)

        assert len(selected) == 10

    def test_mixed_positive_negative(self):
        """Test with mixed positive/negative fitness values."""
        fitness = np.array([
            [10.0, -5.0, 0.0],
            [-5.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ])

        selected = lexicase_selection(fitness, 10, seed=42)

        assert len(selected) == 10

    def test_float32_dtype(self):
        """Test with float32 dtype."""
        fitness = np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        selected = lexicase_selection(fitness, 5, seed=42)

        assert len(selected) == 5

    def test_int_dtype(self):
        """Test with integer dtype."""
        fitness = np.array([
            [10, 0, 10],
            [0, 10, 0],
            [5, 5, 5],
        ], dtype=np.int32)

        selected = lexicase_selection(fitness, 5, seed=42)

        assert len(selected) == 5

    def test_list_input(self):
        """Test with list input (should be converted)."""
        fitness = [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]

        selected = lexicase_selection(fitness, 5, seed=42)

        assert len(selected) == 5


# =============================================================================
# Package Import Tests
# =============================================================================

class TestPackageImports:
    """Test that package imports work correctly."""

    def test_import_lexicase(self):
        """Test basic import works."""
        import lexicase
        assert hasattr(lexicase, 'lexicase_selection')
        assert hasattr(lexicase, 'epsilon_lexicase_selection')
        assert hasattr(lexicase, 'downsample_lexicase_selection')
        assert hasattr(lexicase, 'informed_downsample_lexicase_selection')

    def test_import_numpy_impl(self):
        """Test numpy_impl can be imported."""
        from lexicase import numpy_impl
        assert hasattr(numpy_impl, 'numpy_lexicase_selection')

    def test_version_exists(self):
        """Test version is defined."""
        assert hasattr(lexicase, '__version__')


# =============================================================================
# Stress Tests
# =============================================================================

class TestStress:
    """Stress tests with large data."""

    def test_large_population_many_cases(self):
        """Test with large population and many cases."""
        np.random.seed(42)
        fitness = np.random.rand(500, 100)

        selected = lexicase_selection(fitness, 100, seed=42)

        assert len(selected) == 100
        assert all(0 <= idx < 500 for idx in selected)

    def test_many_selections(self):
        """Test selecting many individuals."""
        np.random.seed(42)
        fitness = np.random.rand(50, 20)

        selected = lexicase_selection(fitness, 1000, seed=42)

        assert len(selected) == 1000

    def test_epsilon_stress(self):
        """Stress test epsilon lexicase."""
        np.random.seed(42)
        fitness = np.random.rand(100, 50)

        selected = epsilon_lexicase_selection(fitness, 50, epsilon=0.1, seed=42)

        assert len(selected) == 50

    def test_downsample_stress(self):
        """Stress test downsampled lexicase."""
        np.random.seed(42)
        fitness = np.random.rand(100, 50)

        selected = downsample_lexicase_selection(fitness, 50, 10, seed=42)

        assert len(selected) == 50

    def test_informed_downsample_stress(self):
        """Stress test informed downsampled lexicase."""
        np.random.seed(42)
        fitness = np.random.rand(100, 50)

        selected = informed_downsample_lexicase_selection(
            fitness, 50, 10, seed=42, sample_rate=0.1
        )

        assert len(selected) == 50
