# API reference

Every selection function takes `(fitness_matrix, num_selected, ...)` and returns
an array of selected indices, with repeats, in the input's backend.

Common parameters:

| Parameter | Meaning |
|---|---|
| `fitness_matrix` | Shape `(n_individuals, n_cases)`. Higher is better |
| `num_selected` | How many parents to return |
| `seed` | Integer. On JAX this may also be a PRNG key |
| `elitism` | Fill the first slots with the best individuals by total fitness |
| `backend` | `"auto"` (default), `"numpy"`, `"jax"`, `"torch"` |

## Selection methods

::: lexicase.lexicase_selection

::: lexicase.epsilon_lexicase_selection

::: lexicase.downsample_lexicase_selection

::: lexicase.informed_downsample_lexicase_selection

::: lexicase.informed_downsample_cases

::: lexicase.batch_lexicase_selection

::: lexicase.cohort_lexicase_selection

::: lexicase.plexicase_selection

::: lexicase.plexicase_probabilities

::: lexicase.dalex_selection

## Backend helpers

::: lexicase.backends
    options:
      members:
        - is_jax_array
        - is_torch_tensor
        - jax_is_available
        - torch_is_available
        - resolve_backend

## NumPy kernels

The functions the NumPy backend dispatches to. They take a
`numpy.random.Generator` instead of a seed and skip validation.

::: lexicase.numpy_impl
    options:
      members:
        - sanitize
        - resolve_pass_threshold
        - numpy_lexicase_selection
        - numpy_epsilon_lexicase_selection
        - numpy_epsilon_lexicase_selection_with_mad
        - numpy_compute_mad_epsilon
        - numpy_downsample_lexicase_selection
        - numpy_informed_downsample_lexicase_selection
        - numpy_batch_lexicase_selection
        - numpy_cohort_lexicase_selection
        - numpy_plexicase_selection
        - numpy_plexicase_probabilities
        - numpy_dalex_selection

## JAX kernels

The jittable and vmappable layer. See [Backends](backends.md) for which arguments
have to be static.

::: lexicase.jax_impl

## Torch kernels

The layer that never synchronizes with the host.

::: lexicase.torch_impl
