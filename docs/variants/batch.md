# Batch lexicase selection

Cases are shuffled and cut into consecutive batches. Each batch filters the
candidate pool on the mean fitness over that batch, rather than on one case at a
time. The batch size is a selection pressure knob: a batch size of 1 is standard
lexicase, and a batch size covering every case is elitist selection on mean
fitness.

!!! quote "Paper"
    Aenugu, S. and Spector, L. (2019). Lexicase Selection in Learning Classifier
    Systems. GECCO '19, pp. 356-364. Algorithm 2.

## When to use it

When lexicase is too strict for your problem. The paper's motivating case is
learning classifier systems, where each rule only covers a sparse subset of the
data, so single-case filtering eliminates most of the population immediately and
selection becomes dominated by whichever case came first. Averaging over a batch
smooths that out.

## The two readings

The paper's pseudocode and its prose specify different survival rules, and both
are available here.

- The pseudocode keeps candidates whose mean fitness on the batch is strictly
  greater than a fixed threshold. Pass `threshold=`. The paper uses 0.9 with
  accuracy-valued fitness in [0, 1].
- The prose says the rules "which are elite on batches of data are allowed to
  survive". That is the default here, with `threshold=None`, because it does not
  assume fitness lies on any particular scale.

With a threshold, a batch that would eliminate every remaining candidate is
skipped instead, which the paper does not specify.

## Usage

```python
from lexicase import batch_lexicase_selection

# elite on each batch
selected = batch_lexicase_selection(fitness, 100, batch_size=10, seed=0)

# the paper's threshold rule, on accuracy-valued fitness
selected = batch_lexicase_selection(accuracy, 100, batch_size=100, seed=0, threshold=0.9)
```

::: lexicase.batch_lexicase_selection
