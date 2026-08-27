# DALex, diversely aggregated lexicase selection

Each selection event draws an importance score per case from a normal
distribution, softmaxes the scores into weights, and takes the individual with the
best weighted mean fitness. That is it: one softmax and one matrix multiply for
every parent at once, no filtering loop at all.

!!! quote "Paper"
    Ni, A., Ding, L., and Spector, L. (2024). DALex: Lexicase-Like Selection via
    Diverse Aggregation. EuroGP 2024, LNCS 14631, pp. 90-107. Algorithm 1.

## Particularity pressure

The standard deviation of the importance scores is the one knob. As it grows, the
softmax concentrates on a single case and the method converges on standard
lexicase. As it shrinks, the weights flatten and the method converges on picking
the best mean fitness.

| Pressure | Behaviour |
|---|---|
| 0 | Elitist selection on mean fitness |
| 3 | What the paper uses for symbolic regression |
| 20 | The default here, and what the paper uses for program synthesis and LCS |
| 200 | Close enough to standard lexicase that the test suite compares them directly |

`relaxed=True` standardizes each case before aggregating, which is how the paper
emulates epsilon lexicase. Use it when your cases are on different scales.

## When to use it

When selection itself is your bottleneck. On a GPU it is not close: at 2000
individuals and 500 cases it is 0.2 ms on CUDA against 88 ms for full lexicase.
See [benchmarks](../benchmarks.md).

## Tie-breaking

Ties in the weighted sum go to the lowest index, which is what `argmax` does. With
continuous weights that only matters when two individuals have identical fitness
vectors.

## Usage

```python
from lexicase import dalex_selection

selected = dalex_selection(fitness, 100, seed=0)
selected = dalex_selection(fitness, 100, seed=0, particularity_pressure=200.0)
selected = dalex_selection(-errors, 100, seed=0, particularity_pressure=3.0, relaxed=True)
```

::: lexicase.dalex_selection
