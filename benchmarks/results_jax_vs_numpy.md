# NumPy against JAX

Milliseconds per call, median of 5 timed calls after one warmup call that pays for compilation. Each call selects as many parents as there are individuals, from a fitness matrix of integers in `{0, 1, 2, 3}`.

```
cpu: AMD Ryzen 7 9800X3D 8-Core Processor
os: Linux x86_64
python: 3.12.11
numpy: 2.5.2
jax: 0.11.1 on NVIDIA GeForce RTX 5080
```

## lexicase

| individuals x cases | numpy | jax (cpu) | jax (gpu) | best JAX speedup |
|---|---|---|---|---|
| 100 x 50 | 1.04 | 38.06 | 55.85 | 0.03x |
| 250 x 100 | 3.07 | 42.56 | 69.30 | 0.07x |
| 500 x 200 | 7.72 | 65.61 | 92.30 | 0.12x |
| 1000 x 300 | 19.88 | 155.90 | 88.67 | 0.22x |
| 2000 x 500 | 56.24 | 774.69 | 116.03 | 0.48x |

## epsilon (MAD)

| individuals x cases | numpy | jax (cpu) | jax (gpu) | best JAX speedup |
|---|---|---|---|---|
| 100 x 50 | 2.28 | 36.99 | 47.30 | 0.06x |
| 250 x 100 | 6.73 | 41.10 | 48.34 | 0.16x |
| 500 x 200 | 17.14 | 65.82 | 54.05 | 0.32x |
| 1000 x 300 | 44.71 | 126.53 | 67.43 | 0.66x |
| 2000 x 500 | 124.86 | 609.65 | 113.98 | 1.10x |

## downsample 10%

| individuals x cases | numpy | jax (cpu) | jax (gpu) | best JAX speedup |
|---|---|---|---|---|
| 100 x 50 | 1.24 | 35.35 | 56.01 | 0.03x |
| 250 x 100 | 3.55 | 40.97 | 68.26 | 0.09x |
| 500 x 200 | 8.51 | 46.06 | 93.04 | 0.18x |
| 1000 x 300 | 20.87 | 50.46 | 83.37 | 0.41x |
| 2000 x 500 | 56.62 | 120.01 | 83.38 | 0.68x |

## dalex

| individuals x cases | numpy | jax (cpu) | jax (gpu) | best JAX speedup |
|---|---|---|---|---|
| 100 x 50 | 0.08 | 0.45 | 0.44 | 0.19x |
| 250 x 100 | 0.37 | 0.69 | 0.37 | 1.00x |
| 500 x 200 | 1.74 | 0.85 | 0.44 | 3.95x |
| 1000 x 300 | 7.02 | 1.46 | 0.42 | 16.67x |
| 2000 x 500 | 30.03 | 5.04 | 0.52 | 57.36x |

