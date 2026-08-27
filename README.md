# lexicase

[![CI](https://github.com/ryanboldi/lexicase/actions/workflows/ci.yml/badge.svg)](https://github.com/ryanboldi/lexicase/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/lexicase.svg)](https://pypi.org/project/lexicase/)
[![Python versions](https://img.shields.io/pypi/pyversions/lexicase.svg)](https://pypi.org/project/lexicase/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Fast, vectorized lexicase selection and its variants, in NumPy, JAX, and PyTorch.

Lexicase selection picks a parent by taking the test cases in a random order and,
one case at a time, throwing out every candidate that is not best on that case,
until one candidate is left. Because the case order is redrawn for every selection
event, an individual that is the best in the population on a single hard case gets
selected as often as a well-rounded individual that is best on none, which is how
lexicase keeps specialists alive where an aggregating method would discard them.
That property is why it is the default parent selection method for program
synthesis, and why it does well on any problem where every case has to be passed
at once.

Documentation: **https://ryanboldi.github.io/lexicase/**

## Install

```bash
pip install lexicase                # numpy only
pip install "lexicase[jax]"         # adds the JAX backend
pip install "lexicase[torch]"       # adds the Torch backend
```

`import lexicase` never imports jax or torch, even when they are installed. The
backend is chosen from the type of the array you pass in.

## Quickstart

```python
import numpy as np
from lexicase import lexicase_selection

# rows are individuals, columns are test cases, higher is better
fitness = np.array([
    [10, 5, 8],
    [8, 9, 6],
    [6, 7, 9],
    [4, 3, 7],
])

selected = lexicase_selection(fitness, num_selected=5, seed=42)
print(selected)
# [2 0 1 2 2]
```

Errors instead of fitness? Negate them. Everything in this package assumes higher
is better.

```python
selected = lexicase_selection(-errors, num_selected=100, seed=0)
```

## Which variant should I use

| If | Use | Why |
|---|---|---|
| Your cases are pass/fail or integer counts | `lexicase_selection` | Exact ties are meaningful, so plain filtering works |
| Your cases are continuous errors | `epsilon_lexicase_selection` | Exact ties never happen with floats, so plain lexicase degenerates to "best on the first case" |
| Evaluation is the bottleneck | `downsample_lexicase_selection` | Uses a random subset of cases per event, so you can run more generations for the same evaluation budget |
| Evaluation is the bottleneck and your cases are redundant | `informed_downsample_lexicase_selection` | Picks a subset whose cases disagree with each other, instead of a uniform sample |
| You want the same subsampling saving but every case used somewhere | `cohort_lexicase_selection` | Splits the population and the cases into paired cohorts |
| Lexicase is too strict and kills your population's diversity | `batch_lexicase_selection` | Filters on the mean over a batch of cases, so batch size tunes selection pressure |
| Selection is the bottleneck, not evaluation | `dalex_selection` | One softmax and one matrix multiply. Fastest thing here by a wide margin |
| You want the selection distribution itself, not samples | `plexicase_probabilities` | Returns the approximate per-individual selection probability |
| Some cases matter more than others | `case_weights=` on `lexicase_selection` | Weighted case ordering instead of a uniform shuffle |
| You want to keep the best individual no matter what | `elitism=` on any of them | Fills the first slots by total fitness |

Default recommendation: `epsilon_lexicase_selection` for regression and anything
continuous, `lexicase_selection` for program synthesis and anything discrete,
and add `downsample_size` once evaluation cost starts hurting.

## Backends

Backend follows the input. A NumPy array gives a NumPy array back, a JAX array
gives a JAX array, a Torch tensor gives a Torch tensor on the same device.
`backend="numpy"`, `"jax"`, or `"torch"` forces it.

```python
import jax.numpy as jnp
import torch
from lexicase import lexicase_selection

lexicase_selection(jnp.asarray(fitness), 100, seed=0)          # jax array out
lexicase_selection(torch.tensor(fitness, device="cuda"), 100, seed=0)  # cuda tensor out
lexicase_selection(fitness, 100, seed=0, backend="jax")        # forced
```

| Backend | Native kernels | Notes |
|---|---|---|
| NumPy | all | The default. Stops filtering as soon as one candidate is left, which makes it hard to beat on CPU |
| JAX | all except `plexicase_selection` | Static shapes throughout, so `lexicase.jax_impl.*` is jittable and vmappable |
| Torch | all except `plexicase_selection` | Preserves device and dtype. No host synchronization anywhere in the selection path |

### JAX

The public `lexicase.*` functions validate their arguments in Python, so they are
not jittable. The kernels in `lexicase.jax_impl` are. Mark the size arguments
static:

```python
import functools, jax
from lexicase.jax_impl import jax_lexicase_selection

select = jax.jit(jax_lexicase_selection, static_argnums=(1,))
parents = select(fitness, 100, jax.random.PRNGKey(0))

# and vmap over a batch of populations
batched = jax.vmap(functools.partial(jax_lexicase_selection, num_selected=100))
parents = batched(populations, key=jax.random.split(key, len(populations)))
```

`num_selected`, `downsample_size`, `batch_size`, `num_cohorts`, `elitism`, and
`mode` are static. Everything else can be traced. `examples/05_jax_jitted_ga.py`
runs a whole GA, selection included, as one `lax.scan` inside one `jit`, with 64
independent runs `vmap`ped together.

### Torch

The Torch backend exists for using lexicase as a selection operator inside a
trainer, where the fitness matrix is a per-sample per-objective reward tensor
already sitting on the GPU. No kernel in it synchronizes with the host: no
`.item()`, no `.cpu()`, no Python branching on tensor values. Selection events
are batched and the per-case loop's trip count comes from the tensor's shape,
never from its contents.

The cost of that is no early exit: the loop always walks every case, even once
every event has narrowed to one candidate. That is the right trade on an
accelerator and the wrong one on CPU, which the benchmark table below shows.

`plexicase_selection` is the exception. Finding the Pareto set boundaries needs a
data-dependent number of candidates, so it runs the NumPy kernel on the host and
moves the result back to your device. It does synchronize, and its docstring says
so.

`benchmarks/torch_cuda_check.py` runs every kernel under
`torch.cuda.set_sync_debug_mode("error")` and reports whether any of them stalled.
On an RTX 5080 with torch 2.11.0+cu128, all thirteen call shapes run clean.

## Variants

Every variant is implemented from a published algorithm, and its docstring cites
the paper, the section, and the algorithm or equation it follows.

| Function | Paper |
|---|---|
| `lexicase_selection` | Helmuth, Spector, and Matheson (2015) |
| `epsilon_lexicase_selection` | La Cava, Helmuth, Spector, and Moore (2019), Algorithms 2, 3, and 4 |
| `downsample_lexicase_selection` | Hernandez, Lalejini, Dolson, and Ofria (2019) |
| `informed_downsample_lexicase_selection` | Boldi et al. (2024) |
| `batch_lexicase_selection` | Aenugu and Spector (2019), Algorithm 2 |
| `cohort_lexicase_selection` | Hernandez, Lalejini, Dolson, and Ofria (2019), Section 4 |
| `plexicase_selection`, `plexicase_probabilities` | Ding, Pantridge, and Spector (2023), Equations 1 to 4 |
| `dalex_selection` | Ni, Ding, and Spector (2024), Algorithm 1 |

### Epsilon modes

`epsilon_lexicase_selection(..., mode=...)` implements all three variants from La
Cava et al. (2019):

- `"static"`: the elite and epsilon both come from the whole population, and the
  fitness matrix is reduced to pass or fail once per call
- `"semi-dynamic"` (default): epsilon comes from the population, the elite comes
  from the current candidate pool. The paper's own recommended default
- `"dynamic"`: both come from the current pool, so an explicit epsilon is not used

With `epsilon=None` the tolerance is the median absolute deviation of each case,
which is the parameter-free version from the paper.

### Batch lexicase

Aenugu and Spector's pseudocode filters on an absolute fitness threshold, while
their prose says candidates that are "elite on batches of data" survive. Both are
available: pass `threshold=` for the pseudocode, leave it out for the prose
version, which is the one that does not assume fitness lies on a particular scale.

## API

Every selection function takes `(fitness_matrix, num_selected, ...)` and returns
an array of selected indices, with repeats, in the input's backend.

| Parameter | Applies to | Meaning |
|---|---|---|
| `fitness_matrix` | all | Shape `(n_individuals, n_cases)`. Higher is better |
| `num_selected` | all | How many parents to return |
| `seed` | all | Integer. On JAX this may also be a PRNG key |
| `elitism` | all | Fill the first slots with the best individuals by total fitness |
| `backend` | all | `"auto"` (default), `"numpy"`, `"jax"`, `"torch"` |
| `epsilon` | epsilon, plexicase | Scalar or per-case tolerance. `None` means MAD |
| `mode` | epsilon | `"static"`, `"semi-dynamic"`, `"dynamic"` |
| `case_weights` | lexicase, epsilon | Positive weight per case for a non-uniform case order |
| `downsample_size` | downsample, informed | Cases kept |
| `sample_rate`, `threshold` | informed | Population fraction used to estimate solve patterns, and the pass or fail cutoff |
| `batch_size`, `threshold` | batch | Cases per batch, and the optional absolute survival threshold |
| `num_cohorts` | cohort | Number of population and case cohorts |
| `alpha` | plexicase | Temperature on the selection distribution |
| `particularity_pressure`, `relaxed` | dalex | Importance score spread, and whether to standardize cases first |

Full generated reference: https://ryanboldi.github.io/lexicase/api/

### NaN

A NaN fitness value is treated as the worst possible performance on its case, on
every backend. An individual that is NaN on a case loses that case to anyone who
scored a number there, and ties with anyone else who is NaN. An individual that is
NaN everywhere is never selected as long as any other individual exists. This is
one elementwise pass, which is what lets the rule be the same on GPU as it is on
CPU. Infinities are left alone, so `-inf` is a usable way to say "failed".

## Benchmarks

`benchmarks/bench.py` produces this table. Milliseconds per call, median of 3
timed calls after one warmup call that pays for compilation. Each call selects as
many parents as there are individuals, from a fitness matrix of integers in
`{0, 1, 2, 3}`, so ties are common and the filtering has real work to do.

```
cpu: AMD Ryzen 7 9800X3D 8-Core Processor
os: Linux 6.12.10-76061203-generic
python: 3.12.11
numpy: 2.5.2
jax: 0.11.1 on NVIDIA GeForce RTX 5080
torch: 2.11.0+cu128 on NVIDIA GeForce RTX 5080
```

1000 individuals, 200 cases:

| method | numpy | jax (cpu) | jax (gpu) | torch (cpu) | torch (cuda) |
|---|---|---|---|---|---|
| lexicase | 19.4 | 116.1 | 85.1 | 118.3 | **9.3** |
| epsilon (MAD) | 42.9 | 94.5 | 60.2 | 109.0 | **10.0** |
| downsample 10% | 20.7 | 45.6 | 82.5 | 15.0 | **1.0** |
| plexicase | 66.8 | 66.8 | 65.7 | 65.9 | 65.8 |
| dalex | 3.4 | 1.4 | 0.5 | 1.1 | **0.1** |

2000 individuals, 500 cases:

| method | numpy | jax (cpu) | jax (gpu) | torch (cpu) | torch (cuda) |
|---|---|---|---|---|---|
| lexicase | **55.5** | 788.3 | 112.8 | 1514.0 | 88.3 |
| epsilon (MAD) | **124.0** | 614.7 | 113.9 | 981.9 | 90.7 |
| downsample 10% | 56.1 | 117.5 | 81.4 | 192.9 | **8.9** |
| plexicase | **559.7** | 562.9 | 571.0 | 576.7 | 565.5 |
| dalex | 28.0 | 4.9 | 0.5 | 6.5 | **0.2** |

`benchmarks/results.md` has all four sizes. Reading it:

- On CPU, NumPy is the backend to use. It stops filtering the moment one
  candidate is left, and that early exit beats a batched kernel that has to walk
  every case.
- On a GPU, the batched kernels win wherever the work is wide. The gap is largest
  for `dalex_selection`, which is one matrix multiply.
- Full lexicase at 2000 by 500 is the one place NumPy still beats CUDA, because
  by then the early exit is saving most of the work.
- `plexicase` has no accelerator kernel, so its row is flat: it runs on the host
  either way.

Do not read these as a general claim about JAX or Torch. They are one machine,
one fitness distribution, and one shape of workload.

## Examples

`examples/` has nine runnable scripts, each seeded and under about 80 lines. See
[examples/README.md](examples/README.md) for the list. Highlights:

- `02_lexicase_vs_tournament.py`: an uncompromising problem where lexicase solves
  18 of 30 runs and tournament selection solves 2
- `05_jax_jitted_ga.py`: a whole GA under one `jit`, 64 runs `vmap`ped together
- `09_torch_rl_selection.py`: selection over a reward tensor that never leaves the GPU

## Tests

```bash
pip install -e ".[dev,jax,torch]"
pytest tests/
```

The suite checks the algorithms, not just that they run. Selection frequencies are
compared against exact probabilities computed by enumerating every case ordering,
the backends are compared against each other distributionally since their RNG
streams cannot match, and Hypothesis covers the invariants over generated
matrices, including degenerate ones.

## Citation

If you use this package, cite the software and the informed down-sampling paper:

```bibtex
@software{lexicase,
  title = {lexicase: fast lexicase selection in NumPy, JAX, and PyTorch},
  author = {Bahlous-Boldi, Ryan},
  url = {https://github.com/ryanboldi/lexicase},
  version = {0.4.0},
  license = {MIT},
}

@article{boldi2024informed,
  title = {Informed Down-Sampled Lexicase Selection: Identifying Productive
           Training Cases for Efficient Problem Solving},
  author = {Boldi, Ryan and Briesch, Martin and Sobania, Dominik and Lalejini,
            Alexander and Helmuth, Thomas and Rothlauf, Franz and Ofria, Charles
            and Spector, Lee},
  journal = {Evolutionary Computation},
  volume = {32},
  number = {4},
  pages = {307--337},
  year = {2024},
  doi = {10.1162/evco_a_00346},
}
```

`CITATION.cff` carries the same information in machine-readable form.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md). New variants are welcome as long as they come
from a published algorithm you can point at, and land with tests.

## References

- Spector, L. (2012). Assessment of Problem Modality by Differential Performance of
  Lexicase Selection in Genetic Programming: A Preliminary Report. GECCO '12
  Companion, pp. 401-408.
- Helmuth, T., Spector, L., and Matheson, J. (2015). Solving Uncompromising
  Problems with Lexicase Selection. IEEE Transactions on Evolutionary Computation
  19(5), 630-643.
- La Cava, W., Spector, L., and Danai, K. (2016). Epsilon-Lexicase Selection for
  Regression. GECCO '16, pp. 741-748.
- Aenugu, S. and Spector, L. (2019). Lexicase Selection in Learning Classifier
  Systems. GECCO '19, pp. 356-364.
- Hernandez, J. G., Lalejini, A., Dolson, E., and Ofria, C. (2019). Random
  Subsampling Improves Performance in Lexicase Selection. GECCO '19 Companion,
  pp. 2028-2031.
- La Cava, W., Helmuth, T., Spector, L., and Moore, J. H. (2019). A Probabilistic
  and Multi-Objective Analysis of Lexicase Selection and Epsilon-Lexicase
  Selection. Evolutionary Computation 27(3), 377-402.
- Ding, L., Pantridge, E., and Spector, L. (2023). Probabilistic Lexicase
  Selection. GECCO '23, pp. 1073-1081.
- Boldi, R., Briesch, M., Sobania, D., Lalejini, A., Helmuth, T., Rothlauf, F.,
  Ofria, C., and Spector, L. (2024). Informed Down-Sampled Lexicase Selection:
  Identifying Productive Training Cases for Efficient Problem Solving.
  Evolutionary Computation 32(4), 307-337.
- Ni, A., Ding, L., and Spector, L. (2024). DALex: Lexicase-Like Selection via
  Diverse Aggregation. EuroGP 2024, LNCS 14631, pp. 90-107.

## License

MIT. See [LICENSE](LICENSE).
