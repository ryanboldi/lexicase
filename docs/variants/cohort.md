# Cohort lexicase selection

Randomly split the population into K cohorts and the case set into K cohorts, then
pair them up. Population cohort k competes only within itself, judged only by case
cohort k. Every case is used somewhere, but each individual is only ever compared
on 1/K of them.

!!! quote "Paper"
    Hernandez, J. G., Lalejini, A., Dolson, E., and Ofria, C. (2019). Random
    Subsampling Improves Performance in Lexicase Selection. GECCO '19 Companion,
    pp. 2028-2031. Section 4.

## When to use it

Same motivation as [downsampling](downsample.md): cut the per-generation
evaluation count by a factor of K. The difference is that downsampling throws away
K-1 of every K cases for that generation, while cohorts keep using all of them,
just on different individuals.

The paper describes it as an island model where both the membership and the
environment are randomized every generation. Which of the two subsampling schemes
works better depends on the problem, and the paper does not find a general winner.

## Implementation note

Cohorts are drawn once per call, so a single call with many `num_selected` gives
lumpy selection shares: only cohort members ever compete with each other. That is
the algorithm, not an artifact. In a generational loop you call it once per
generation and the cohorts are redrawn each time.

The selections are spread across cohorts as evenly as the count allows, and are
returned grouped by cohort.

## Usage

```python
from lexicase import cohort_lexicase_selection

selected = cohort_lexicase_selection(fitness, 100, num_cohorts=4, seed=0)
```

::: lexicase.cohort_lexicase_selection
