# Downsampled lexicase selection

Each selection event uses a random subset of the cases instead of all of them.
Fewer cases per event means less work per event, and in a real evolutionary run it
means you only have to evaluate individuals on the sampled cases, which is usually
where the actual cost is.

!!! quote "Paper"
    Hernandez, J. G., Lalejini, A., Dolson, E., and Ofria, C. (2019). Random
    Subsampling Improves Performance in Lexicase Selection. GECCO '19 Companion,
    pp. 2028-2031.

    Follow-up: Helmuth, T. and Spector, L. (2022). Problem-Solving Benefits of
    Down-Sampled Lexicase Selection. *Artificial Life* 27(3-4), 183-203.

## When to use it

Whenever evaluating an individual on a case costs something. Downsampling by a
factor of D divides the per-generation evaluation budget by D, which you spend on
more generations or a bigger population. The papers find that trade is usually
worth making, and often improves solve rates outright rather than just matching
them.

It also raises diversity, because different selection events see different cases.

Going too small has a cost. `examples/03_downsample_vs_full.py` shows the shape of
it on a 100-case problem: downsampling to 25 cases matches full lexicase's solve
rate at less than half the selection time, and downsampling to 5 drops from 14 out
of 15 runs solved to 11.

## Implementation note

The subset is drawn per selection event, not once per call. One draw of a
permutation truncated to `downsample_size` gives both the subset and its order, so
there is no separate shuffle.

## Usage

```python
from lexicase import downsample_lexicase_selection

# 10 of however many cases there are, redrawn for each of the 100 events
selected = downsample_lexicase_selection(fitness, 100, downsample_size=10, seed=0)
```

::: lexicase.downsample_lexicase_selection
