"""
Property-based tests for the selection invariants.

These check the rules that have to hold for every fitness matrix, not just the
handful in the example-based tests: index ranges, output length, what lexicase
is allowed to select, how the variants relate to each other, and what happens
on degenerate input.
"""

import itertools
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import lexicase
from lexicase import (
    batch_lexicase_selection,
    cohort_lexicase_selection,
    dalex_selection,
    downsample_lexicase_selection,
    epsilon_lexicase_selection,
    lexicase_selection,
    plexicase_probabilities,
    plexicase_selection,
)

SLOW = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
DEFAULT = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def fitness_matrices(draw, max_individuals=6, max_cases=5, allow_nan=False):
    """Matrices with plenty of ties, since ties are where selection gets interesting."""
    n_individuals = draw(st.integers(1, max_individuals))
    n_cases = draw(st.integers(1, max_cases))
    coarse = draw(st.booleans())
    if coarse:
        elements = st.integers(0, 3).map(float)
    else:
        elements = st.floats(
            min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False, width=32
        )
    if allow_nan:
        elements = st.one_of(elements, st.just(float("nan")))
    return draw(arrays(np.float64, (n_individuals, n_cases), elements=elements))


def exact_probabilities(fitness):
    """Selection probability of every individual, over all case orderings."""
    n_individuals, n_cases = fitness.shape
    totals = np.zeros(n_individuals)
    for order in itertools.permutations(range(n_cases)):
        candidates = np.arange(n_individuals)
        for case in order:
            if len(candidates) <= 1:
                break
            values = fitness[candidates, case]
            candidates = candidates[values == values.max()]
        totals[candidates] += 1.0 / len(candidates)
    return totals / math.factorial(n_cases)


def frequencies(indices, n_individuals):
    counts = np.bincount(np.asarray(indices), minlength=n_individuals).astype(float)
    return counts / counts.sum()


def total_variation(a, b):
    return 0.5 * np.abs(a - b).sum()


METHODS = {
    "lexicase": lambda f, n, **kw: lexicase_selection(f, n, seed=0, **kw),
    "epsilon": lambda f, n, **kw: epsilon_lexicase_selection(f, n, seed=0, **kw),
    "epsilon-static": lambda f, n, **kw: epsilon_lexicase_selection(
        f, n, seed=0, mode="static", **kw
    ),
    "epsilon-dynamic": lambda f, n, **kw: epsilon_lexicase_selection(
        f, n, seed=0, mode="dynamic", **kw
    ),
    "downsample": lambda f, n, **kw: downsample_lexicase_selection(
        f, n, max(1, f.shape[1] // 2), seed=0, **kw
    ),
    "batch": lambda f, n, **kw: batch_lexicase_selection(
        f, n, max(1, f.shape[1] // 2), seed=0, **kw
    ),
    "cohort": lambda f, n, **kw: cohort_lexicase_selection(
        f, n, min(2, f.shape[0], f.shape[1]), seed=0, **kw
    ),
    "plexicase": lambda f, n, **kw: plexicase_selection(f, n, seed=0, **kw),
    "dalex": lambda f, n, **kw: dalex_selection(f, n, seed=0, **kw),
}


class TestUniversalInvariants:
    @pytest.mark.parametrize("name", sorted(METHODS))
    @given(fitness=fitness_matrices(), num_selected=st.integers(0, 12))
    @DEFAULT
    def test_indices_are_in_range_and_output_has_the_right_length(
        self, name, fitness, num_selected
    ):
        selected = np.asarray(METHODS[name](fitness, num_selected))
        assert len(selected) == num_selected
        assert np.all(selected >= 0)
        assert np.all(selected < fitness.shape[0])

    @pytest.mark.parametrize("name", sorted(METHODS))
    @given(fitness=fitness_matrices(), num_selected=st.integers(1, 8))
    @DEFAULT
    def test_same_seed_gives_the_same_output(self, name, fitness, num_selected):
        first = np.asarray(METHODS[name](fitness, num_selected))
        second = np.asarray(METHODS[name](fitness, num_selected))
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("name", sorted(METHODS))
    @given(fitness=fitness_matrices(), data=st.data())
    @DEFAULT
    def test_elites_always_appear(self, name, fitness, data):
        n_individuals = fitness.shape[0]
        elitism = data.draw(st.integers(1, n_individuals))
        num_selected = data.draw(st.integers(elitism, elitism + 5))

        selected = np.asarray(METHODS[name](fitness, num_selected, elitism=elitism))
        totals = fitness.sum(axis=1)
        expected = np.sort(totals)[-elitism:]
        # Ties make which index is elite ambiguous, so compare the values.
        observed = np.sort(totals[selected[:elitism]])
        np.testing.assert_allclose(observed, expected)


class TestLexicaseSpecific:
    @given(fitness=fitness_matrices(), num_selected=st.integers(1, 10))
    @DEFAULT
    def test_every_selected_individual_is_elite_on_some_case(self, fitness, num_selected):
        selected = lexicase_selection(fitness, num_selected, seed=0)
        elite = fitness >= fitness.max(axis=0, keepdims=True)
        assert np.all(elite[selected].any(axis=1))

    @given(fitness=fitness_matrices(), num_selected=st.integers(0, 10))
    @DEFAULT
    def test_zero_epsilon_reproduces_lexicase_exactly(self, fitness, num_selected):
        plain = lexicase_selection(fitness, num_selected, seed=17)
        with_epsilon = epsilon_lexicase_selection(fitness, num_selected, 0.0, seed=17)
        np.testing.assert_array_equal(plain, with_epsilon)

    @given(fitness=fitness_matrices(max_individuals=5, max_cases=3))
    @SLOW
    def test_full_downsample_matches_lexicase_distributionally(self, fitness):
        expected = exact_probabilities(fitness)
        observed = frequencies(
            downsample_lexicase_selection(
                fitness, 4000, fitness.shape[1], seed=1
            ),
            fitness.shape[0],
        )
        assert total_variation(expected, observed) < 0.06

    @given(fitness=fitness_matrices(max_individuals=5, max_cases=3), data=st.data())
    @SLOW
    def test_duplicating_an_individual_only_moves_probability_mass(self, fitness, data):
        twin = data.draw(st.integers(0, fitness.shape[0] - 1))
        original = exact_probabilities(fitness)
        extended = exact_probabilities(np.vstack([fitness, fitness[twin]]))

        # The copy is as selectable as its twin, and they now split what one had.
        np.testing.assert_allclose(extended[-1], extended[twin], atol=1e-12)
        # Nobody who could be selected before becomes unselectable, and nobody new
        # becomes selectable.
        np.testing.assert_array_equal(original > 0, extended[:-1] > 0)
        np.testing.assert_allclose(extended.sum(), 1.0)


class TestDegenerateInput:
    @given(num_selected=st.integers(1, 6), n_cases=st.integers(1, 5))
    @DEFAULT
    def test_single_individual_is_always_the_answer(self, num_selected, n_cases):
        fitness = np.arange(n_cases, dtype=float).reshape(1, n_cases)
        for method in METHODS.values():
            selected = np.asarray(method(fitness, num_selected))
            assert np.all(selected == 0)

    @given(fitness=fitness_matrices(max_cases=1), num_selected=st.integers(1, 8))
    @DEFAULT
    def test_single_case_selects_only_from_the_elites_of_that_case(
        self, fitness, num_selected
    ):
        selected = lexicase_selection(fitness, num_selected, seed=0)
        best = fitness[:, 0].max()
        assert np.all(fitness[selected, 0] == best)

    @given(
        n_individuals=st.integers(1, 6),
        n_cases=st.integers(1, 5),
        value=st.floats(-10, 10, width=32),
        num_selected=st.integers(1, 20),
    )
    @DEFAULT
    def test_all_equal_fitness_selects_uniformly_at_random(
        self, n_individuals, n_cases, value, num_selected
    ):
        fitness = np.full((n_individuals, n_cases), value)
        selected = np.asarray(lexicase_selection(fitness, num_selected, seed=0))
        assert np.all(selected >= 0)
        assert np.all(selected < n_individuals)
        probabilities = plexicase_probabilities(fitness)
        np.testing.assert_allclose(probabilities, 1.0 / n_individuals)

    @given(fitness=fitness_matrices(), num_selected=st.integers(1, 8))
    @DEFAULT
    def test_a_row_of_nan_is_never_selected_when_a_real_row_exists(
        self, fitness, num_selected
    ):
        with_nan = np.vstack([fitness, np.full(fitness.shape[1], np.nan)])
        selected = np.asarray(lexicase_selection(with_nan, num_selected, seed=0))
        assert fitness.shape[0] not in set(selected.tolist())

    @given(fitness=fitness_matrices(allow_nan=True), num_selected=st.integers(0, 8))
    @DEFAULT
    def test_nan_never_crashes_and_never_leaks_into_the_result(
        self, fitness, num_selected
    ):
        selected = np.asarray(lexicase_selection(fitness, num_selected, seed=0))
        assert len(selected) == num_selected
        assert np.all(selected >= 0)
        assert np.all(selected < fitness.shape[0])

    @given(fitness=fitness_matrices())
    @DEFAULT
    def test_plexicase_probabilities_are_a_distribution(self, fitness):
        probabilities = plexicase_probabilities(fitness)
        assert np.all(probabilities >= 0)
        np.testing.assert_allclose(probabilities.sum(), 1.0)


class TestOtherBackends:
    @given(fitness=fitness_matrices(), num_selected=st.integers(0, 8))
    @SLOW
    def test_jax_indices_are_in_range_and_reproducible(self, fitness, num_selected):
        jnp = pytest.importorskip("jax.numpy")
        matrix = jnp.asarray(fitness)
        first = np.asarray(lexicase_selection(matrix, num_selected, seed=4))
        second = np.asarray(lexicase_selection(matrix, num_selected, seed=4))
        np.testing.assert_array_equal(first, second)
        assert len(first) == num_selected
        assert np.all((first >= 0) & (first < fitness.shape[0]))

    @given(fitness=fitness_matrices(), num_selected=st.integers(0, 8))
    @SLOW
    def test_torch_indices_are_in_range_and_reproducible(self, fitness, num_selected):
        torch = pytest.importorskip("torch")
        matrix = torch.as_tensor(fitness)
        first = np.asarray(lexicase_selection(matrix, num_selected, seed=4))
        second = np.asarray(lexicase_selection(matrix, num_selected, seed=4))
        np.testing.assert_array_equal(first, second)
        assert len(first) == num_selected
        assert np.all((first >= 0) & (first < fitness.shape[0]))

    @given(fitness=fitness_matrices(), num_selected=st.integers(1, 6))
    @SLOW
    def test_every_backend_only_selects_case_elites(self, fitness, num_selected):
        elite = fitness >= fitness.max(axis=0, keepdims=True)
        for backend in ("numpy", "jax", "torch"):
            if backend == "jax" and not lexicase.jax_is_available():
                continue
            if backend == "torch" and not lexicase.torch_is_available():
                continue
            selected = np.asarray(
                lexicase_selection(fitness, num_selected, seed=0, backend=backend)
            )
            assert np.all(elite[selected].any(axis=1))
