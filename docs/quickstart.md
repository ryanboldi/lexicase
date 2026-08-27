# Quickstart

## The fitness matrix

Everything takes one array of shape `(n_individuals, n_cases)` where **higher is
better**. Rows are individuals, columns are test cases.

```python
import numpy as np
from lexicase import lexicase_selection

fitness = np.array([
    [10, 5, 8],   # individual 0
    [8, 9, 6],    # individual 1
    [6, 7, 9],    # individual 2
    [4, 3, 7],    # individual 3
])

selected = lexicase_selection(fitness, num_selected=5, seed=42)
print(selected)
# [2 0 1 2 2]
```

The return value is an array of indices into the population, with repeats. Those
are your parents. Individual 3 never appears, because it is not the best on any
case, and standard lexicase can only ever return an individual that is elite on
at least one case.

## If you have errors, not fitness

Negate them.

```python
selected = lexicase_selection(-errors, num_selected=100, seed=0)
```

## Continuous fitness needs epsilon

With floating point errors, exact ties essentially never happen, so plain lexicase
collapses to "whoever is best on the first case drawn". Epsilon lexicase keeps
everyone within a tolerance of the best.

```python
from lexicase import epsilon_lexicase_selection

# tolerance defaults to the median absolute deviation of each case
selected = epsilon_lexicase_selection(-errors, num_selected=100, seed=0)
```

See [epsilon lexicase](variants/epsilon.md) for the three modes.

## Elitism

Any method can reserve the first slots for the best individuals by total fitness.

```python
selected = lexicase_selection(fitness, num_selected=100, seed=0, elitism=2)
```

## Seeds

Pass `seed=` for anything you want to reproduce. The same seed gives the same
result on the same backend, every time. It does not give the same result *across*
backends, because NumPy, JAX, and Torch have different random number generators
and there is no way to make their index sequences agree. What does hold across
backends is the selection distribution, and the test suite checks that against
exact probabilities computed by enumerating every case ordering.

## A generation loop

```python
import numpy as np
from lexicase import epsilon_lexicase_selection

rng = np.random.default_rng(0)
population = rng.normal(size=(200, 4))

for generation in range(50):
    errors = evaluate(population)              # (200, n_cases)
    parents = epsilon_lexicase_selection(
        -errors, num_selected=200, seed=generation, elitism=1
    )
    population = mutate(population[parents], rng)
```

`examples/` has nine complete versions of this, including a JAX one that runs the
whole loop under `jit` and a Torch one that keeps everything on the GPU.
