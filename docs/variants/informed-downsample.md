# Informed downsampled lexicase selection

Instead of sampling cases uniformly, pick a subset whose cases disagree with each
other. Two cases that every individual either passes together or fails together
carry the same information, and spending two downsample slots on them wastes one.

The subset is chosen by farthest first traversal over Hamming distances between
the cases' solve patterns, estimated from a random sample of the population.

!!! quote "Paper"
    Boldi, R., Briesch, M., Sobania, D., Lalejini, A., Helmuth, T., Rothlauf, F.,
    Ofria, C., and Spector, L. (2024). Informed Down-Sampled Lexicase Selection:
    Identifying Productive Training Cases for Efficient Problem Solving.
    *Evolutionary Computation* 32(4), 307-337.

## When to use it

When you already want [downsampling](downsample.md) and your case set has
redundancy in it, which most benchmark suites do. If your cases are all
independent, uniform sampling is already doing the right thing and this only adds
cost.

`sample_rate` controls how much of the population is used to estimate the solve
patterns. It defaults to 0.01, which is cheap. Raise it if your population is
small enough that 1% is one or two individuals.

`examples/04_informed_downsampling.py` builds twelve cases in three redundant
groups of four and picks three of them. Informed downsampling covers all three
groups in 63% of trials against uniform sampling's 30%.

## Implementation note

The case subset is chosen once per call and reused for every selection event,
which matches how it is used in a generational loop: sample once per generation,
evaluate on that subset, select. All three backends agree on this.

## Usage

```python
from lexicase import informed_downsample_lexicase_selection

selected = informed_downsample_lexicase_selection(
    fitness, 100, downsample_size=10, seed=0, sample_rate=0.01
)
```

::: lexicase.informed_downsample_lexicase_selection
