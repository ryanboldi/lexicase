"""Tests for the lexicase variants added in 0.4."""

import itertools
import math

import numpy as np
import pytest

from lexicase import (
    batch_lexicase_selection,
    cohort_lexicase_selection,
    dalex_selection,
    epsilon_lexicase_selection,
    lexicase_selection,
    plexicase_probabilities,
    plexicase_selection,
)

FITNESS = np.array(
    [
        [4.0, 1.0, 2.0],
        [1.0, 4.0, 2.0],
        [2.0, 2.0, 4.0],
        [4.0, 1.0, 1.0],
        [3.0, 3.0, 3.0],
    ]
)


def enumerate_probabilities(fitness, mode=None, epsilon=None):
    """Exact selection probabilities over every case ordering."""
    n_individuals, n_cases = fitness.shape
    matrix = fitness
    if mode == "static":
        case_best = fitness.max(axis=0)
        matrix = (fitness >= case_best - epsilon).astype(float)

    totals = np.zeros(n_individuals)
    for order in itertools.permutations(range(n_cases)):
        candidates = np.arange(n_individuals)
        for case in order:
            if len(candidates) <= 1:
                break
            values = matrix[candidates, case]
            if mode == "dynamic":
                tolerance = np.median(np.abs(values - np.median(values)))
            elif mode == "semi-dynamic":
                tolerance = epsilon[case]
            else:
                tolerance = 0.0
            candidates = candidates[values >= values.max() - tolerance]
        totals[candidates] += 1.0 / len(candidates)
    return totals / math.factorial(n_cases)


def frequencies(indices, n_individuals):
    counts = np.bincount(np.asarray(indices), minlength=n_individuals).astype(float)
    return counts / counts.sum()


def total_variation(a, b):
    return 0.5 * np.abs(a - b).sum()


class TestEpsilonModes:
    @pytest.mark.parametrize("mode", ["static", "semi-dynamic"])
    def test_matches_exact_probabilities(self, mode):
        epsilon = np.array([1.0, 1.0, 1.0])
        expected = enumerate_probabilities(FITNESS, mode, epsilon)
        observed = frequencies(
            epsilon_lexicase_selection(FITNESS, 40000, epsilon, seed=1, mode=mode),
            len(FITNESS),
        )
        assert total_variation(expected, observed) < 0.02

    def test_dynamic_matches_exact_probabilities(self):
        expected = enumerate_probabilities(FITNESS, "dynamic")
        observed = frequencies(
            epsilon_lexicase_selection(FITNESS, 40000, seed=1, mode="dynamic"),
            len(FITNESS),
        )
        assert total_variation(expected, observed) < 0.02

    def test_semi_dynamic_with_zero_epsilon_is_standard_lexicase(self):
        expected = enumerate_probabilities(FITNESS)
        observed = frequencies(
            epsilon_lexicase_selection(FITNESS, 40000, 0.0, seed=2),
            len(FITNESS),
        )
        assert total_variation(expected, observed) < 0.02

    def test_static_with_zero_epsilon_still_loses_within_pass_resolution(self):
        # Static mode replaces the errors with pass/fail against the population
        # elite, so individuals 0 and 3 become indistinguishable even though
        # standard lexicase separates them on case 2.
        zero = np.zeros(FITNESS.shape[1])
        expected = enumerate_probabilities(FITNESS, "static", zero)
        observed = frequencies(
            epsilon_lexicase_selection(FITNESS, 40000, 0.0, seed=2, mode="static"),
            len(FITNESS),
        )
        assert total_variation(expected, observed) < 0.02
        assert expected[3] > 0
        assert enumerate_probabilities(FITNESS)[3] == 0

    def test_static_is_stricter_than_semi_dynamic(self):
        epsilon = np.array([1.0, 1.0, 1.0])
        static = enumerate_probabilities(FITNESS, "static", epsilon)
        semi = enumerate_probabilities(FITNESS, "semi-dynamic", epsilon)
        assert set(np.flatnonzero(static)) <= set(np.flatnonzero(semi))

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown epsilon mode"):
            epsilon_lexicase_selection(FITNESS, 5, seed=0, mode="wobbly")

    def test_dynamic_rejects_explicit_epsilon(self):
        with pytest.raises(ValueError, match="mode='dynamic'"):
            epsilon_lexicase_selection(FITNESS, 5, 0.5, seed=0, mode="dynamic")


class TestCaseWeights:
    def test_first_case_frequency_is_proportional_to_weight(self):
        from lexicase.numpy_impl import _case_order

        weights = np.array([3.0, 1.0, 1.0])
        rng = np.random.default_rng(3)
        firsts = np.array([_case_order(3, rng, weights)[0] for _ in range(30000)])
        observed = np.bincount(firsts, minlength=3) / len(firsts)
        np.testing.assert_allclose(observed, weights / weights.sum(), atol=0.01)

    def test_heavy_case_dominates_the_ordering(self):
        weights = np.array([1000.0, 1.0, 1.0])
        selected = np.asarray(
            lexicase_selection(FITNESS, 4000, seed=3, case_weights=weights)
        )
        # Case 0 leads roughly 1000/1002 of the time, and its elites are 0 and 3.
        assert np.mean(np.isin(selected, [0, 3])) > 0.99

    def test_uniform_weights_match_uniform_shuffling(self):
        expected = enumerate_probabilities(FITNESS)
        observed = frequencies(
            lexicase_selection(
                FITNESS, 40000, seed=4, case_weights=np.ones(FITNESS.shape[1])
            ),
            len(FITNESS),
        )
        assert total_variation(expected, observed) < 0.02

    def test_rejects_bad_weights(self):
        with pytest.raises(ValueError, match="must have length"):
            lexicase_selection(FITNESS, 5, seed=0, case_weights=[1.0, 1.0])
        with pytest.raises(ValueError, match="must be positive"):
            lexicase_selection(FITNESS, 5, seed=0, case_weights=[1.0, 0.0, 1.0])


class TestBatchLexicase:
    def test_one_case_per_batch_reduces_to_standard_lexicase(self):
        expected = enumerate_probabilities(FITNESS)
        observed = frequencies(
            batch_lexicase_selection(FITNESS, 40000, 1, seed=5), len(FITNESS)
        )
        assert total_variation(expected, observed) < 0.02

    def test_one_batch_is_elitist_on_mean_fitness(self):
        selected = batch_lexicase_selection(FITNESS, 20, FITNESS.shape[1], seed=6)
        best = int(np.argmax(FITNESS.mean(axis=1)))
        assert np.all(np.asarray(selected) == best)

    def test_threshold_filters_on_an_absolute_level(self):
        selected = batch_lexicase_selection(
            FITNESS, 200, FITNESS.shape[1], seed=7, threshold=2.5
        )
        means = FITNESS.mean(axis=1)
        assert np.all(means[np.asarray(selected)] > 2.5)

    def test_impossible_threshold_keeps_the_pool_intact(self):
        selected = batch_lexicase_selection(FITNESS, 200, 1, seed=8, threshold=1e9)
        assert len(np.unique(selected)) == len(FITNESS)

    def test_rejects_bad_batch_size(self):
        with pytest.raises(ValueError, match="Batch size must be positive"):
            batch_lexicase_selection(FITNESS, 5, 0, seed=0)


class TestCohortLexicase:
    def test_one_cohort_reduces_to_standard_lexicase(self):
        expected = enumerate_probabilities(FITNESS)
        observed = frequencies(
            cohort_lexicase_selection(FITNESS, 40000, 1, seed=9), len(FITNESS)
        )
        assert total_variation(expected, observed) < 0.02

    def test_cohorts_partition_the_population(self):
        population = np.random.default_rng(0).random((8, 4))
        for seed in range(20):
            selected = np.asarray(cohort_lexicase_selection(population, 2, 2, seed=seed))
            assert selected[0] != selected[1]

    def test_selection_count_is_exact(self):
        population = np.random.default_rng(0).random((9, 6))
        for num_selected in (1, 5, 7, 20):
            selected = cohort_lexicase_selection(population, num_selected, 3, seed=1)
            assert len(selected) == num_selected

    def test_rejects_too_many_cohorts(self):
        with pytest.raises(ValueError, match="cannot exceed number of cases"):
            cohort_lexicase_selection(FITNESS, 5, 4, seed=0)
        with pytest.raises(ValueError, match="cannot exceed number of individuals"):
            cohort_lexicase_selection(FITNESS[:2], 5, 3, seed=0)


class TestPlexicase:
    def test_probabilities_match_a_hand_computed_example(self):
        # Three specialists, each elite on exactly one case.
        fitness = np.array([[9.0, 0.0], [0.0, 9.0], [1.0, 1.0]])
        probabilities = plexicase_probabilities(fitness)
        np.testing.assert_allclose(probabilities, [0.5, 0.5, 0.0])

    def test_dominated_individuals_get_zero_probability(self):
        fitness = np.array([[5.0, 3.0], [5.0, 1.0], [1.0, 3.0]])
        probabilities = plexicase_probabilities(fitness)
        assert probabilities[0] == 1.0
        assert probabilities[1] == 0.0
        assert probabilities[2] == 0.0

    def test_identical_rows_are_both_kept(self):
        fitness = np.array([[5.0, 1.0], [5.0, 1.0], [1.0, 5.0]])
        probabilities = plexicase_probabilities(fitness)
        assert probabilities[0] > 0 and probabilities[1] > 0
        np.testing.assert_allclose(probabilities[0], probabilities[1])

    def test_alpha_zero_is_uniform_over_the_pareto_boundaries(self):
        probabilities = plexicase_probabilities(FITNESS, alpha=0.0)
        nonzero = probabilities[probabilities > 0]
        np.testing.assert_allclose(nonzero, nonzero[0])

    def test_large_alpha_concentrates_on_the_most_elite(self):
        fitness = np.array([[9.0, 9.0, 0.0, 0.0], [9.0, 0.0, 9.0, 0.0], [0.0, 0.0, 0.0, 9.0]])
        base = plexicase_probabilities(fitness)
        np.testing.assert_allclose(base, [0.375, 0.375, 0.25])
        sharp = plexicase_probabilities(fitness, alpha=8.0)
        assert sharp.max() > base.max()
        assert np.argmax(sharp) == np.argmax(base)

    def test_probabilities_sum_to_one(self):
        population = np.random.default_rng(0).integers(0, 3, size=(40, 12)).astype(float)
        probabilities = plexicase_probabilities(population)
        np.testing.assert_allclose(probabilities.sum(), 1.0)

    def test_sampling_follows_the_computed_distribution(self):
        expected = plexicase_probabilities(FITNESS)
        observed = frequencies(
            plexicase_selection(FITNESS, 40000, seed=10), len(FITNESS)
        )
        assert total_variation(expected, observed) < 0.02

    def test_epsilon_relaxation_widens_the_support(self):
        population = np.random.default_rng(2).random((30, 6))
        strict = plexicase_probabilities(population)
        relaxed = plexicase_probabilities(population, epsilon=0.2)
        assert (relaxed > 0).sum() >= (strict > 0).sum()

    def test_rejects_negative_alpha(self):
        with pytest.raises(ValueError, match="Alpha must be non-negative"):
            plexicase_probabilities(FITNESS, alpha=-1.0)


class TestDalex:
    def test_zero_pressure_is_elitist_on_mean_fitness(self):
        selected = dalex_selection(FITNESS, 20, seed=11, particularity_pressure=0.0)
        best = int(np.argmax(FITNESS.mean(axis=1)))
        assert np.all(np.asarray(selected) == best)

    def test_high_pressure_approaches_lexicase(self):
        expected = enumerate_probabilities(FITNESS)
        observed = frequencies(
            dalex_selection(FITNESS, 40000, seed=12, particularity_pressure=200.0),
            len(FITNESS),
        )
        assert total_variation(expected, observed) < 0.05

    def test_relaxed_standardizes_cases(self):
        scaled = FITNESS * np.array([1.0, 1000.0, 1.0])
        plain = frequencies(
            dalex_selection(scaled, 4000, seed=13, particularity_pressure=1.0),
            len(FITNESS),
        )
        relaxed = frequencies(
            dalex_selection(
                scaled, 4000, seed=13, particularity_pressure=1.0, relaxed=True
            ),
            len(FITNESS),
        )
        assert total_variation(plain, relaxed) > 0.1

    def test_rejects_negative_pressure(self):
        with pytest.raises(ValueError, match="Particularity pressure"):
            dalex_selection(FITNESS, 5, seed=0, particularity_pressure=-1.0)


class TestElitismOnVariants:
    @pytest.mark.parametrize(
        "call",
        [
            lambda f, **kw: batch_lexicase_selection(f, 6, 2, seed=0, **kw),
            lambda f, **kw: cohort_lexicase_selection(f, 6, 2, seed=0, **kw),
            lambda f, **kw: plexicase_selection(f, 6, seed=0, **kw),
            lambda f, **kw: dalex_selection(f, 6, seed=0, **kw),
        ],
    )
    def test_elites_are_always_present(self, call):
        selected = np.asarray(call(FITNESS, elitism=2))
        assert len(selected) == 6
        elites = set(np.argsort(FITNESS.sum(axis=1))[-2:].tolist())
        assert elites <= set(selected.tolist())
