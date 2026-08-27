# Benchmarks

`benchmarks/bench.py` produces these numbers. Milliseconds per call, median of 3
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

## 100 individuals, 50 cases

| method | numpy | jax (cpu) | jax (gpu) | torch (cpu) | torch (cuda) |
|---|---|---|---|---|---|
| lexicase | **1.0** | 36.0 | 52.1 | 1.6 | 2.0 |
| epsilon (MAD) | 2.3 | 36.3 | 47.0 | **2.0** | 2.2 |
| downsample 10% | 1.3 | 35.8 | 53.9 | **0.3** | **0.3** |
| plexicase | **0.5** | 0.7 | 0.6 | **0.5** | **0.5** |
| dalex | **0.1** | 0.5 | 0.5 | **0.1** | **0.1** |

## 500 individuals, 100 cases

| method | numpy | jax (cpu) | jax (gpu) | torch (cpu) | torch (cuda) |
|---|---|---|---|---|---|
| lexicase | 7.2 | 58.4 | 92.7 | 20.0 | **3.9** |
| epsilon (MAD) | 15.8 | 52.6 | 52.9 | 18.7 | **4.4** |
| downsample 10% | 8.3 | 43.1 | 91.7 | 3.6 | **0.6** |
| plexicase | 13.1 | 13.0 | 13.1 | **12.9** | **12.9** |
| dalex | 0.8 | 0.7 | 0.6 | 0.3 | **0.1** |

## 1000 individuals, 200 cases

| method | numpy | jax (cpu) | jax (gpu) | torch (cpu) | torch (cuda) |
|---|---|---|---|---|---|
| lexicase | 19.4 | 116.1 | 85.1 | 118.3 | **9.3** |
| epsilon (MAD) | 42.9 | 94.5 | 60.2 | 109.0 | **10.0** |
| downsample 10% | 20.7 | 45.6 | 82.5 | 15.0 | **1.0** |
| plexicase | 66.8 | 66.8 | **65.7** | 65.9 | 65.8 |
| dalex | 3.4 | 1.4 | 0.5 | 1.1 | **0.1** |

## 2000 individuals, 500 cases

| method | numpy | jax (cpu) | jax (gpu) | torch (cpu) | torch (cuda) |
|---|---|---|---|---|---|
| lexicase | **55.5** | 788.3 | 112.8 | 1514.0 | 88.3 |
| epsilon (MAD) | **124.0** | 614.7 | 113.9 | 981.9 | 90.7 |
| downsample 10% | 56.1 | 117.5 | 81.4 | 192.9 | **8.9** |
| plexicase | **559.7** | 562.9 | 571.0 | 576.7 | 565.5 |
| dalex | 28.0 | 4.9 | 0.5 | 6.5 | **0.2** |

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
