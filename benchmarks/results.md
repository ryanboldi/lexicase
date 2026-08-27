# Benchmark results

Milliseconds per call, median of 3 timed calls after one warmup call. Each call selects as many parents as there are individuals.

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
| lexicase | 1.0 | 36.0 | 52.1 | 1.6 | 2.0 |
| epsilon (MAD) | 2.3 | 36.3 | 47.0 | 2.0 | 2.2 |
| downsample 10% | 1.3 | 35.8 | 53.9 | 0.3 | 0.3 |
| plexicase | 0.5 | 0.7 | 0.6 | 0.5 | 0.5 |
| dalex | 0.1 | 0.5 | 0.5 | 0.1 | 0.1 |

## 500 individuals, 100 cases

| method | numpy | jax (cpu) | jax (gpu) | torch (cpu) | torch (cuda) |
|---|---|---|---|---|---|
| lexicase | 7.2 | 58.4 | 92.7 | 20.0 | 3.9 |
| epsilon (MAD) | 15.8 | 52.6 | 52.9 | 18.7 | 4.4 |
| downsample 10% | 8.3 | 43.1 | 91.7 | 3.6 | 0.6 |
| plexicase | 13.1 | 13.0 | 13.1 | 12.9 | 12.9 |
| dalex | 0.8 | 0.7 | 0.6 | 0.3 | 0.1 |

## 1000 individuals, 200 cases

| method | numpy | jax (cpu) | jax (gpu) | torch (cpu) | torch (cuda) |
|---|---|---|---|---|---|
| lexicase | 19.4 | 116.1 | 85.1 | 118.3 | 9.3 |
| epsilon (MAD) | 42.9 | 94.5 | 60.2 | 109.0 | 10.0 |
| downsample 10% | 20.7 | 45.6 | 82.5 | 15.0 | 1.0 |
| plexicase | 66.8 | 66.8 | 65.7 | 65.9 | 65.8 |
| dalex | 3.4 | 1.4 | 0.5 | 1.1 | 0.1 |

## 2000 individuals, 500 cases

| method | numpy | jax (cpu) | jax (gpu) | torch (cpu) | torch (cuda) |
|---|---|---|---|---|---|
| lexicase | 55.5 | 788.3 | 112.8 | 1514.0 | 88.3 |
| epsilon (MAD) | 124.0 | 614.7 | 113.9 | 981.9 | 90.7 |
| downsample 10% | 56.1 | 117.5 | 81.4 | 192.9 | 8.9 |
| plexicase | 559.7 | 562.9 | 571.0 | 576.7 | 565.5 |
| dalex | 28.0 | 4.9 | 0.5 | 6.5 | 0.2 |

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
