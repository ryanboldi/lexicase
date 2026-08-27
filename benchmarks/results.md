# Benchmark results

Milliseconds per call, median of 3 timed calls after one warmup call. Each call selects as many parents as there are individuals.

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
| lexicase | 1.0 | 34.5 | 53.7 | 1.7 | 2.0 |
| epsilon (MAD) | 2.3 | 37.1 | 49.5 | 2.6 | 2.2 |
| downsample 10% | 1.3 | 35.0 | 55.3 | 0.4 | 0.3 |
| plexicase | 0.5 | 0.7 | 0.7 | 0.5 | 0.5 |
| dalex | 0.1 | 0.5 | 0.7 | 0.1 | 0.1 |

## 500 individuals, 100 cases

| method | numpy | jax (cpu) | jax (gpu) | torch (cpu) | torch (cuda) |
|---|---|---|---|---|---|
| lexicase | 7.4 | 53.3 | 91.5 | 15.0 | 4.0 |
| epsilon (MAD) | 15.5 | 51.8 | 54.6 | 15.9 | 4.4 |
| downsample 10% | 8.1 | 44.4 | 92.5 | 2.8 | 0.5 |
| plexicase | 12.8 | 13.4 | 13.5 | 13.2 | 13.1 |
| dalex | 0.8 | 0.9 | 0.5 | 0.3 | 0.1 |

## 1000 individuals, 200 cases

| method | numpy | jax (cpu) | jax (gpu) | torch (cpu) | torch (cuda) |
|---|---|---|---|---|---|
| lexicase | 19.2 | 113.7 | 85.4 | 110.4 | 9.3 |
| epsilon (MAD) | 41.8 | 99.1 | 61.3 | 122.4 | 10.0 |
| downsample 10% | 20.9 | 50.0 | 89.8 | 15.4 | 1.0 |
| plexicase | 65.3 | 65.3 | 64.7 | 64.8 | 65.2 |
| dalex | 3.8 | 1.4 | 0.6 | 1.5 | 0.1 |

## 2000 individuals, 500 cases

| method | numpy | jax (cpu) | jax (gpu) | torch (cpu) | torch (cuda) |
|---|---|---|---|---|---|
| lexicase | 56.5 | 792.3 | 113.5 | 1612.0 | 88.3 |
| epsilon (MAD) | 122.6 | 626.3 | 112.5 | 1635.2 | 90.7 |
| downsample 10% | 56.4 | 120.1 | 81.1 | 176.1 | 9.0 |
| plexicase | 577.3 | 582.1 | 581.4 | 581.0 | 569.9 |
| dalex | 27.3 | 4.9 | 1.0 | 10.8 | 0.2 |

Notes:

- The NumPy kernel stops filtering the moment one candidate is left. The JAX
  and Torch kernels batch every selection event and always walk every case,
  because checking for an early exit means reading a value, which breaks jit
  on JAX and stalls the pipeline on Torch. That is why NumPy stays competitive
  on CPU and why the accelerator backends win when the work is wide.
- `plexicase` builds one distribution and samples from it, so its cost barely
  moves with the number of parents drawn. It has no native JAX or Torch kernel,
  so its row is flat across backends: it runs on the host either way.
- `dalex` is one softmax and one matrix multiply, which is why it is the
  fastest row everywhere and the widest gap in favour of an accelerator.
- `skipped` means the first call took longer than the budget passed to
  --budget, and nothing else.
