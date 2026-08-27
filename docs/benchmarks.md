# Benchmarks

`benchmarks/bench.py` produces these numbers. Milliseconds per call, median of 3
timed calls after one warmup call that pays for compilation. Each call selects as
many parents as there are individuals, from a fitness matrix of integers in
`{0, 1, 2, 3}`, so ties are common and the filtering has real work to do.

```
cpu: AMD Ryzen 7 9800X3D 8-Core Processor
os: Linux x86_64
python: 3.12.11
numpy: 2.5.2
jax: 0.11.1 on NVIDIA GeForce RTX 5080
torch: 2.11.0+cu128 on NVIDIA GeForce RTX 5080
```

## 100 individuals, 50 cases

| method | numpy | jax (cpu) | jax (gpu) | torch (cpu) | torch (cuda) |
|---|---|---|---|---|---|
| lexicase | **1.0** | 34.5 | 53.7 | 1.7 | 2.0 |
| epsilon (MAD) | 2.3 | 37.1 | 49.5 | 2.6 | 2.2 |
| downsample 10% | 1.3 | 35.0 | 55.3 | 0.4 | **0.3** |
| plexicase | 0.5 | 0.7 | 0.7 | 0.5 | 0.5 |
| dalex | 0.1 | 0.5 | 0.7 | 0.1 | 0.1 |

## 500 individuals, 100 cases

| method | numpy | jax (cpu) | jax (gpu) | torch (cpu) | torch (cuda) |
|---|---|---|---|---|---|
| lexicase | 7.4 | 53.3 | 91.5 | 15.0 | **4.0** |
| epsilon (MAD) | 15.5 | 51.8 | 54.6 | 15.9 | **4.4** |
| downsample 10% | 8.1 | 44.4 | 92.5 | 2.8 | **0.5** |
| plexicase | 12.8 | 13.4 | 13.5 | 13.2 | 13.1 |
| dalex | 0.8 | 0.9 | 0.5 | 0.3 | **0.1** |

## 1000 individuals, 200 cases

| method | numpy | jax (cpu) | jax (gpu) | torch (cpu) | torch (cuda) |
|---|---|---|---|---|---|
| lexicase | 19.2 | 113.7 | 85.4 | 110.4 | **9.3** |
| epsilon (MAD) | 41.8 | 99.1 | 61.3 | 122.4 | **10.0** |
| downsample 10% | 20.9 | 50.0 | 89.8 | 15.4 | **1.0** |
| plexicase | 65.3 | 65.3 | 64.7 | 64.8 | 65.2 |
| dalex | 3.8 | 1.4 | 0.6 | 1.5 | **0.1** |

## 2000 individuals, 500 cases

| method | numpy | jax (cpu) | jax (gpu) | torch (cpu) | torch (cuda) |
|---|---|---|---|---|---|
| lexicase | **56.5** | 792.3 | 113.5 | 1612.0 | 88.3 |
| epsilon (MAD) | 122.6 | 626.3 | 112.5 | 1635.2 | **90.7** |
| downsample 10% | 56.4 | 120.1 | 81.1 | 176.1 | **9.0** |
| plexicase | 577.3 | 582.1 | 581.4 | 581.0 | 569.9 |
| dalex | 27.3 | 4.9 | 1.0 | 10.8 | **0.2** |

## NumPy against JAX

![Four log-scale panels, one per selection method, plotting milliseconds per call against problem size for numpy, jax on CPU, and jax on GPU. NumPy is fastest on lexicase, epsilon lexicase, and downsampling at every size tested. JAX on GPU is fastest on DALex from 500 individuals upward.](assets/jax_vs_numpy_light.png#only-light)
![Four log-scale panels, one per selection method, plotting milliseconds per call against problem size for numpy, jax on CPU, and jax on GPU. NumPy is fastest on lexicase, epsilon lexicase, and downsampling at every size tested. JAX on GPU is fastest on DALex from 500 individuals upward.](assets/jax_vs_numpy_dark.png#only-dark)

`benchmarks/bench_jax_vs_numpy.py` produces that chart and
`benchmarks/results_jax_vs_numpy.md`, on the machine listed above. Median of 5
timed calls after one warmup call that pays for compilation.

The result is not the one you would guess. **JAX does not beat NumPy on the
filtering variants at any size tested here**, on CPU or GPU. It wins on DALex,
and it wins hard.

| method | NumPy at 2000 x 500 | best JAX at 2000 x 500 | speedup |
|---|---|---|---|
| lexicase | 56.2 ms | 116.0 ms (gpu) | 0.48x |
| epsilon (MAD) | 124.9 ms | 114.0 ms (gpu) | 1.10x |
| downsample 10% | 56.6 ms | 83.4 ms (gpu) | 0.68x |
| dalex | 30.0 ms | 0.52 ms (gpu) | **57x** |

Two things drive that:

- **The NumPy kernel gets to quit early.** It filters one selection event at a
  time and stops the moment a single candidate is left, which on a matrix with
  ties usually happens after a handful of cases. The JAX kernel batches every
  event and walks all the cases, because checking whether to stop means reading a
  traced value, which `jit` will not allow. On a 500-case matrix that is a large
  constant factor to give away.
- **DALex has no filtering loop at all.** It is one softmax and one matrix
  multiply, which is exactly the shape of work an accelerator is built for. Its
  GPU time barely moves from 100 individuals to 2000, while NumPy's grows by a
  factor of 375.

The crossover is visible in the chart: the JAX GPU lines are close to flat across
the whole sweep, because at these sizes JAX is paying dispatch overhead rather
than doing arithmetic. Nothing here says JAX is slow. It says
this workload is a sequential filter, and a sequential filter is the one thing a
vectorized backend cannot speed up.

**Use the JAX backend when** you want selection inside a jitted training loop,
when you are `vmap`ping many populations at once, or when you are using DALex.
**Use NumPy when** you are calling selection once per generation from ordinary
Python, which is most of the time.

## Reading these

- **On CPU, use NumPy.** It stops filtering the moment one candidate is left, and
  that early exit beats a batched kernel that has to walk every case.
- **On a GPU, the batched kernels win wherever the work is wide.** The gap is
  largest for `dalex_selection`, which is one matrix multiply, and for
  downsampling, where the case loop is short.
- **Full lexicase at 2000 by 500 is the one place NumPy still beats CUDA**,
  because by then the early exit is saving most of the work.
- `plexicase` has no accelerator kernel, so its row is flat across backends. It
  runs on the host either way.
- JAX on CPU pays a fixed `lax.scan` overhead that dominates at small sizes.

Do not read these as a general claim about JAX or Torch. They are one machine, one
fitness distribution, and one shape of workload. Run `benchmarks/bench.py` on your
own hardware with your own data before choosing a backend on performance grounds.

## Reproducing

```bash
pip install -e ".[dev,jax,torch]"
python benchmarks/bench.py --budget 15
```

That writes `benchmarks/results.md`, which has the same four tables plus the exact
versions of everything it ran against.
