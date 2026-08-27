# Backends

The backend follows the type of array you pass in.

| You pass | You get back |
|---|---|
| NumPy array, list, anything `np.asarray` handles | NumPy array |
| JAX array | JAX array |
| Torch tensor | Torch tensor on the same device |

`backend="numpy"`, `"jax"`, or `"torch"` overrides that.

```python
import jax.numpy as jnp
import torch
from lexicase import lexicase_selection

lexicase_selection(numpy_fitness, 100, seed=0)                        # numpy out
lexicase_selection(jnp.asarray(fitness), 100, seed=0)                 # jax out
lexicase_selection(torch.tensor(fitness, device="cuda"), 100, seed=0) # cuda tensor out
lexicase_selection(numpy_fitness, 100, seed=0, backend="jax")         # forced
```

## Importing never pulls in a backend

`import lexicase` does not import jax or torch, even when they are installed. A
JAX array or a Torch tensor cannot exist unless the caller has already imported
that library, so detection reads `sys.modules` instead of importing anything:

```python
def is_jax_array(x):
    jax = sys.modules.get("jax")
    array_type = getattr(jax, "Array", None)
    return array_type is not None and isinstance(x, array_type)
```

`lexicase.jax_is_available()` and `lexicase.torch_is_available()` do import, and
say so in their docstrings.

## Seeds across backends

The same seed gives the same result on the same backend, every time. It does not
give the same result across backends. NumPy's Generator, JAX's counter-based PRNG,
and Torch's Generator produce different streams and there is no way to make their
index sequences agree.

What does hold across backends is the *distribution*. The test suite checks that
two ways: each backend's empirical selection frequencies are compared by
chi-square against exact probabilities computed by enumerating every case
ordering, and the backends are compared against each other by total variation
distance. The tests are named for what they check, for instance
`test_numpy_and_jax_lexicase_are_distributionally_equivalent`.

## NumPy

The default, and the one to use on CPU. It filters one selection event at a time
and stops the moment a single candidate is left, which is a large saving that the
batched accelerator kernels cannot take.

## JAX

Every method except [plexicase](variants/plexicase.md) has a native kernel in
`lexicase.jax_impl`. They use static shapes throughout: `lax.scan` over the case
order, a boolean candidate mask, and `vmap` over selection events.

The public `lexicase.*` functions validate their arguments in Python, so **they
are not jittable**. The kernels in `lexicase.jax_impl` are:

```python
import functools
import jax
from lexicase.jax_impl import jax_lexicase_selection

select = jax.jit(jax_lexicase_selection, static_argnums=(1,))
parents = select(fitness, 100, jax.random.PRNGKey(0))

batched = jax.vmap(functools.partial(jax_lexicase_selection, num_selected=100))
parents = batched(populations, key=jax.random.split(key, len(populations)))
```

Static arguments, per function:

| Function | Static |
|---|---|
| `jax_lexicase_selection` | `num_selected`, `elitism` |
| `jax_epsilon_lexicase_selection` | `num_selected`, `elitism`, `mode` |
| `jax_downsample_lexicase_selection` | `num_selected`, `downsample_size`, `elitism` |
| `jax_informed_downsample_lexicase_selection` | `num_selected`, `downsample_size`, `sample_rate`, `elitism` |
| `jax_batch_lexicase_selection` | `num_selected`, `batch_size`, `threshold`, `elitism` |
| `jax_cohort_lexicase_selection` | `num_selected`, `num_cohorts`, `elitism` |
| `jax_dalex_selection` | `num_selected`, `relaxed`, `elitism` |

`jax_batch_lexicase_selection` unrolls its batch loop at trace time, so a large
case count with a small batch size makes compilation slow.

`examples/05_jax_jitted_ga.py` runs a whole GA, selection included, as one
`lax.scan` inside one `jit`, with 64 independent runs `vmap`ped together.

## Torch

Built for using lexicase as a selection operator inside a trainer, where the
fitness matrix is a per-sample per-objective reward tensor already on the GPU.

**No kernel in `lexicase.torch_impl` synchronizes with the host.** No `.item()`,
no `.cpu()`, no Python branching on tensor values. Selection events are batched
and the per-case loop's trip count comes from the tensor's shape, never from its
contents.

The cost is that there is no early exit: the loop always walks every case, even
once every event has narrowed to one candidate. That is the right trade on an
accelerator and the wrong one on CPU, which the [benchmarks](benchmarks.md) show
plainly.

```python
import torch
from lexicase import lexicase_selection

rewards = rollout()                        # (n_samples, n_objectives) on cuda
parents = lexicase_selection(rewards, n_samples, seed=step)
assert parents.device == rewards.device    # nothing came back to the host
```

Two things do cost one host-to-device copy per call, and only one, outside the
per-case loop: passing `case_weights`, `epsilon`, or `threshold` as host data.
Pass them as device tensors to avoid even that.

[`plexicase_selection`](variants/plexicase.md) is the exception and its docstring
says so. Finding the Pareto set boundaries needs a data-dependent number of
candidates, so it runs the NumPy kernel on the host and moves the result back to
your device.

### Verifying it yourself

```bash
python benchmarks/torch_cuda_check.py
```

That runs every kernel under `torch.cuda.set_sync_debug_mode("error")`. On an
RTX 5080 with torch 2.11.0+cu128, all thirteen call shapes report `no sync`.

On CPU, where sync debug mode does not apply, the test suite enforces the same
contract by making `Tensor.item`, `.cpu`, `.tolist`, `.numpy`, and `__bool__`
raise, with a negative control so a broken monkeypatch cannot pass silently.

## NaN

A NaN fitness value is treated as the worst possible performance on its case, on
every backend. An individual that is NaN on a case loses that case to anyone who
scored a number there, and ties with anyone else who is NaN. An individual that is
NaN everywhere is never selected as long as any other individual exists.

This rule was chosen because it costs one elementwise pass, which is what lets it
be identical on GPU and CPU. Raising on NaN would mean reading a value back from
the device on every call, which is exactly what the Torch backend exists to avoid.

Infinities are left alone, so `-inf` is a usable way to say "this individual
failed this case outright".
