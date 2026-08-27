# Examples

Every script here is runnable on its own, seeded, and under about 80 lines.

```bash
pip install -e ".[dev,jax,torch,examples]"
python examples/01_symbolic_regression_epsilon.py
```

| Script | What it shows | Needs |
|---|---|---|
| [01_symbolic_regression_epsilon.py](01_symbolic_regression_epsilon.py) | Fitting a polynomial to noisy data with epsilon lexicase | numpy |
| [02_lexicase_vs_tournament.py](02_lexicase_vs_tournament.py) | An uncompromising problem where lexicase solves 18/30 runs and tournament solves 2/30 | numpy |
| [03_downsample_vs_full.py](03_downsample_vs_full.py) | Downsampled versus full lexicase, solve rate against selection wall clock | numpy |
| [04_informed_downsampling.py](04_informed_downsampling.py) | Informed downsampling covering more distinct case groups than uniform sampling | numpy |
| [05_jax_jitted_ga.py](05_jax_jitted_ga.py) | A whole GA under one `jit`, with 64 runs `vmap`ped together | jax |
| [06_deap_adapter.py](06_deap_adapter.py) | A DEAP-shaped `sel_lexicase(individuals, k)` selector | numpy |
| [07_diversity_over_generations.py](07_diversity_over_generations.py) | Diversity collapse under each selection method, with a plot | matplotlib |
| [08_variant_tour.py](08_variant_tour.py) | Every variant on one population, side by side | numpy |
| [09_torch_rl_selection.py](09_torch_rl_selection.py) | Selection over a reward tensor that never leaves the GPU | torch |

`notebooks/` holds the older Jupyter notebooks. They cover the same ground more
slowly and with more prose.

The scripts in `02`, `03`, and `07` import `baselines/` from the repository root
for tournament and fitness proportionate selection. Those are deliberately not
part of the `lexicase` package: they exist so the comparisons have something to
compare against.
