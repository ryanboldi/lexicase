"""Backend resolution and the backend= override."""

import numpy as np
import pytest

import lexicase
from lexicase.backends import resolve_backend

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

FITNESS = np.array([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0], [2.0, 2.0, 3.0]])

CALLS = {
    "lexicase": lambda f, **kw: lexicase.lexicase_selection(f, 6, seed=0, **kw),
    "epsilon": lambda f, **kw: lexicase.epsilon_lexicase_selection(f, 6, seed=0, **kw),
    "downsample": lambda f, **kw: lexicase.downsample_lexicase_selection(
        f, 6, 2, seed=0, **kw
    ),
    "informed": lambda f, **kw: lexicase.informed_downsample_lexicase_selection(
        f, 6, 2, seed=0, sample_rate=0.5, **kw
    ),
    "batch": lambda f, **kw: lexicase.batch_lexicase_selection(f, 6, 2, seed=0, **kw),
    "cohort": lambda f, **kw: lexicase.cohort_lexicase_selection(f, 6, 3, seed=0, **kw),
    "plexicase": lambda f, **kw: lexicase.plexicase_selection(f, 6, seed=0, **kw),
    "dalex": lambda f, **kw: lexicase.dalex_selection(f, 6, seed=0, **kw),
}


def test_resolve_backend_follows_the_input_type():
    assert resolve_backend("auto", FITNESS) == "numpy"
    assert resolve_backend("auto", jnp.asarray(FITNESS)) == "jax"
    assert resolve_backend("numpy", jnp.asarray(FITNESS)) == "numpy"
    assert resolve_backend("jax", FITNESS) == "jax"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        lexicase.lexicase_selection(FITNESS, 3, seed=0, backend="cupy")


@pytest.mark.parametrize("name", sorted(CALLS))
def test_numpy_in_gives_numpy_out(name):
    result = CALLS[name](FITNESS)
    assert isinstance(result, np.ndarray)
    assert not lexicase.is_jax_array(result)
    assert len(result) == 6


@pytest.mark.parametrize("name", sorted(CALLS))
def test_jax_in_gives_jax_out(name):
    result = CALLS[name](jnp.asarray(FITNESS))
    assert lexicase.is_jax_array(result)
    assert len(result) == 6


@pytest.mark.parametrize("name", sorted(CALLS))
def test_backend_override_wins_over_input_type(name):
    forced_jax = CALLS[name](FITNESS, backend="jax")
    forced_numpy = CALLS[name](jnp.asarray(FITNESS), backend="numpy")
    assert lexicase.is_jax_array(forced_jax)
    assert isinstance(forced_numpy, np.ndarray)


@pytest.mark.parametrize("name", sorted(CALLS))
def test_selected_indices_are_in_range(name):
    for matrix in (FITNESS, jnp.asarray(FITNESS)):
        result = np.asarray(CALLS[name](matrix))
        assert result.min() >= 0
        assert result.max() < len(FITNESS)


def test_jax_available_reports_true_here():
    assert lexicase.jax_is_available()


def test_list_input_uses_the_numpy_backend():
    result = lexicase.lexicase_selection(FITNESS.tolist(), 4, seed=0)
    assert isinstance(result, np.ndarray)


def test_jax_kernels_are_jittable():
    import functools

    from lexicase import jax_impl

    matrix = jnp.asarray(FITNESS)
    key = jax.random.PRNGKey(0)

    jitted = jax.jit(jax_impl.jax_lexicase_selection, static_argnums=(1,))
    assert jitted(matrix, 6, key).shape == (6,)

    jitted_epsilon = jax.jit(
        jax_impl.jax_epsilon_lexicase_selection, static_argnums=(1,)
    )
    assert jitted_epsilon(matrix, 6, 0.5, key).shape == (6,)

    jitted_downsample = jax.jit(
        jax_impl.jax_downsample_lexicase_selection, static_argnums=(1, 2)
    )
    assert jitted_downsample(matrix, 6, 2, key).shape == (6,)

    jitted_dalex = jax.jit(jax_impl.jax_dalex_selection, static_argnums=(1,))
    assert jitted_dalex(matrix, 6, key).shape == (6,)

    batched = jax.vmap(
        functools.partial(jax_impl.jax_lexicase_selection, num_selected=4)
    )
    populations = jnp.stack([matrix, matrix + 1.0])
    assert batched(populations, key=jax.random.split(key, 2)).shape == (2, 4)


def test_zero_selections_returns_empty_on_both_backends():
    assert len(lexicase.lexicase_selection(FITNESS, 0, seed=0)) == 0
    assert len(lexicase.lexicase_selection(jnp.asarray(FITNESS), 0, seed=0)) == 0
