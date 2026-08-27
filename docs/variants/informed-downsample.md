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

## What counts as "solved"

The paper defines the distance between two cases as the Hamming distance between
their binary solve vectors, and it assumes cases are scored pass/fail. So the
fitness matrix has to be reduced to pass/fail before any distance exists.

- **A matrix with at most two distinct values is read as pass/fail directly.**
  That covers 0/1 fitness and the negated 0 and -1 errors this package tells you
  to pass in. This is the paper's own setting and reproduces its numbers exactly.
- **Anything else falls back to a per-case median split**, which is a heuristic
  for continuous fitness, not the rule in the paper. It is defined and stable,
  but "solved" means "above the median on this case", which is relative rather
  than absolute.
- **`threshold=` overrides both.** Pass your own cutoff when you know your pass
  mark. Candidates count as solving a case when their fitness is strictly greater
  than the cutoff.

The Torch backend never infers the cutoff and raises if you do not pass one.
Checking would mean reading the tensor's values, which synchronizes with the
host, and not doing that is the whole point of that backend. Pass
`threshold=0.5` for 0/1 rewards.

## Implementing the k schedule

Algorithm 2 in the paper has two knobs for spending less compute: `rho`, the
fraction of parents evaluated on every case, and `k`, recomputing the distance
matrix only every k generations while still re-drawing the down-sample every
generation. `sample_rate` is `rho`. For `k`, use `informed_downsample_cases`,
which is the case-selection half on its own:

```python
from lexicase import informed_downsample_cases, lexicase_selection

distances = None
for generation in range(generations):
    if generation % k == 0:
        cases, distances = informed_downsample_cases(
            fitness, downsample_size, seed=generation, sample_rate=0.01
        )
    else:
        # same distances, fresh farthest first traversal
        cases, _ = informed_downsample_cases(
            fitness, downsample_size, seed=generation, distances=distances
        )
    parents = lexicase_selection(fitness[:, cases], population_size, seed=generation)
```

That is a host-side helper and always returns NumPy arrays.

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

::: lexicase.informed_downsample_cases
