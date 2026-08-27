"""
Cross-backend correctness.

NumPy and JAX have different PRNGs, so the same seed cannot produce the same
index sequence. What must hold instead is that both backends sample from the
same distribution. These tests check that against exact selection
probabilities computed by enumerating every case ordering.
"""

import itertools
import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from lexicase import (  # noqa: E402
    batch_lexicase_selection,
    cohort_lexicase_selection,
    dalex_selection,
    downsample_lexicase_selection,
    epsilon_lexicase_selection,
    informed_downsample_lexicase_selection,
    lexicase_selection,
)

TRIALS = 20000
ALPHA = 0.001

FITNESS = np.array(
    [
        [4.0, 1.0, 2.0],
        [1.0, 4.0, 2.0],
        [2.0, 2.0, 4.0],
        [4.0, 1.0, 1.0],
        [3.0, 3.0, 3.0],
    ]
)


def exact_lexicase_probabilities(fitness, epsilon=None):
    """Selection probability of every individual, over all case orderings."""
    n_individuals, n_cases = fitness.shape
    totals = np.zeros(n_individuals)
    for order in itertools.permutations(range(n_cases)):
        candidates = np.arange(n_individuals)
        for case in order:
            if len(candidates) <= 1:
                break
            values = fitness[candidates, case]
            tolerance = 0.0 if epsilon is None else epsilon[case]
            candidates = candidates[values >= values.max() - tolerance]
        totals[candidates] += 1.0 / len(candidates)
    return totals / math.factorial(n_cases)


def chi_square_pvalue(counts, expected):
    from scipy import stats

    keep = expected > 0
    assert counts[~keep].sum() == 0, "an impossible individual was selected"
    return stats.chisquare(counts[keep], expected[keep]).pvalue


def counts_of(indices, n_individuals):
    return np.bincount(np.asarray(indices), minlength=n_individuals).astype(float)


def total_variation(a, b):
    return 0.5 * np.abs(a / a.sum() - b / b.sum()).sum()


def test_numpy_lexicase_matches_exact_probabilities():
    expected = exact_lexicase_probabilities(FITNESS) * TRIALS
    counts = counts_of(lexicase_selection(FITNESS, TRIALS, seed=7), len(FITNESS))
    assert chi_square_pvalue(counts, expected) > ALPHA


def test_jax_lexicase_matches_exact_probabilities():
    expected = exact_lexicase_probabilities(FITNESS) * TRIALS
    selected = lexicase_selection(jnp.asarray(FITNESS), TRIALS, seed=7)
    counts = counts_of(selected, len(FITNESS))
    assert chi_square_pvalue(counts, expected) > ALPHA


def test_numpy_and_jax_lexicase_are_distributionally_equivalent():
    numpy_counts = counts_of(lexicase_selection(FITNESS, TRIALS, seed=11), len(FITNESS))
    jax_counts = counts_of(
        lexicase_selection(jnp.asarray(FITNESS), TRIALS, seed=11), len(FITNESS)
    )
    assert total_variation(numpy_counts, jax_counts) < 0.02


def test_numpy_and_jax_epsilon_lexicase_are_distributionally_equivalent():
    epsilon = np.array([1.0, 1.0, 1.0])
    expected = exact_lexicase_probabilities(FITNESS, epsilon) * TRIALS

    numpy_counts = counts_of(
        epsilon_lexicase_selection(FITNESS, TRIALS, epsilon, seed=3), len(FITNESS)
    )
    jax_counts = counts_of(
        epsilon_lexicase_selection(jnp.asarray(FITNESS), TRIALS, epsilon, seed=3),
        len(FITNESS),
    )
    assert chi_square_pvalue(numpy_counts, expected) > ALPHA
    assert chi_square_pvalue(jax_counts, expected) > ALPHA
    assert total_variation(numpy_counts, jax_counts) < 0.02


@pytest.mark.parametrize("mode", ["static", "semi-dynamic", "dynamic"])
def test_numpy_and_jax_epsilon_modes_are_distributionally_equivalent(mode):
    kwargs = {} if mode == "dynamic" else {"epsilon": 1.0}
    numpy_counts = counts_of(
        epsilon_lexicase_selection(FITNESS, TRIALS, seed=5, mode=mode, **kwargs),
        len(FITNESS),
    )
    jax_counts = counts_of(
        epsilon_lexicase_selection(
            jnp.asarray(FITNESS), TRIALS, seed=5, mode=mode, **kwargs
        ),
        len(FITNESS),
    )
    assert total_variation(numpy_counts, jax_counts) < 0.03


def test_numpy_and_jax_downsample_are_distributionally_equivalent():
    numpy_counts = counts_of(
        downsample_lexicase_selection(FITNESS, TRIALS, 2, seed=13), len(FITNESS)
    )
    jax_counts = counts_of(
        downsample_lexicase_selection(jnp.asarray(FITNESS), TRIALS, 2, seed=13),
        len(FITNESS),
    )
    assert total_variation(numpy_counts, jax_counts) < 0.02


def test_numpy_and_jax_batch_are_distributionally_equivalent():
    numpy_counts = counts_of(
        batch_lexicase_selection(FITNESS, TRIALS, 2, seed=17), len(FITNESS)
    )
    jax_counts = counts_of(
        batch_lexicase_selection(jnp.asarray(FITNESS), TRIALS, 2, seed=17),
        len(FITNESS),
    )
    assert total_variation(numpy_counts, jax_counts) < 0.02


def test_numpy_and_jax_dalex_are_distributionally_equivalent():
    numpy_counts = counts_of(
        dalex_selection(FITNESS, TRIALS, seed=19, particularity_pressure=5.0),
        len(FITNESS),
    )
    jax_counts = counts_of(
        dalex_selection(
            jnp.asarray(FITNESS), TRIALS, seed=19, particularity_pressure=5.0
        ),
        len(FITNESS),
    )
    assert total_variation(numpy_counts, jax_counts) < 0.02


def test_numpy_and_jax_cohort_use_the_same_cohort_structure():
    population = np.random.default_rng(0).random((12, 6))
    numpy_counts = counts_of(
        cohort_lexicase_selection(population, 2400, 3, seed=23), len(population)
    )
    jax_counts = counts_of(
        cohort_lexicase_selection(jnp.asarray(population), 2400, 3, seed=23),
        len(population),
    )
    assert (numpy_counts > 0).sum() > 0
    assert (jax_counts > 0).sum() > 0


def test_backends_agree_on_which_individuals_are_reachable():
    population = np.random.default_rng(1).integers(0, 2, size=(30, 8)).astype(float)
    numpy_seen = set(np.asarray(lexicase_selection(population, 4000, seed=2)).tolist())
    jax_seen = set(
        np.asarray(lexicase_selection(jnp.asarray(population), 4000, seed=2)).tolist()
    )
    assert numpy_seen == jax_seen


def test_informed_downsample_agrees_across_backends_on_reachable_individuals():
    population = np.random.default_rng(4).random((40, 10))
    numpy_seen = set(
        np.asarray(
            informed_downsample_lexicase_selection(
                population, 2000, 4, seed=6, sample_rate=0.5
            )
        ).tolist()
    )
    jax_seen = set(
        np.asarray(
            informed_downsample_lexicase_selection(
                jnp.asarray(population), 2000, 4, seed=6, sample_rate=0.5
            )
        ).tolist()
    )
    assert numpy_seen and jax_seen
    assert numpy_seen <= set(range(40)) and jax_seen <= set(range(40))


def test_same_seed_is_reproducible_within_each_backend():
    numpy_a = lexicase_selection(FITNESS, 50, seed=99)
    numpy_b = lexicase_selection(FITNESS, 50, seed=99)
    np.testing.assert_array_equal(numpy_a, numpy_b)

    jax_a = lexicase_selection(jnp.asarray(FITNESS), 50, seed=99)
    jax_b = lexicase_selection(jnp.asarray(FITNESS), 50, seed=99)
    np.testing.assert_array_equal(np.asarray(jax_a), np.asarray(jax_b))


def test_jax_accepts_a_prng_key_as_the_seed():
    key = jax.random.PRNGKey(5)
    first = lexicase_selection(jnp.asarray(FITNESS), 20, seed=key)
    second = lexicase_selection(jnp.asarray(FITNESS), 20, seed=key)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
