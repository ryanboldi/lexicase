# Weighted case order

Standard lexicase draws the case order from a uniform shuffle. `case_weights=`
replaces that with weighted sampling without replacement, so a case with twice the
weight is twice as likely to come first, and the same holds recursively down the
rest of the order.

This uses the exponential key scheme of Efraimidis and Spirakis (2006): draw
$u_c \sim \text{Uniform}(0,1)$ for each case, compute $u_c^{1/w_c}$, and sort
descending.

## When to use it

When you have prior knowledge that some cases matter more, and you want that to
show up as selection pressure rather than as a change to the fitness values. A
case with a high weight will lead the order most of the time, so only individuals
elite on it survive most events.

Weights must be strictly positive. There is no published lexicase variant this
comes from; it is the obvious generalization of the uniform shuffle.

## Usage

```python
import numpy as np
from lexicase import epsilon_lexicase_selection, lexicase_selection

weights = np.ones(n_cases)
weights[hard_cases] = 5.0

selected = lexicase_selection(fitness, 100, seed=0, case_weights=weights)
selected = epsilon_lexicase_selection(fitness, 100, seed=0, case_weights=weights)
```

With weights `[3, 1, 1]`, case 0 leads 3/5 of the time. `examples/08_variant_tour.py`
shows what that does to the selection shares.

## Reference

Efraimidis, P. S. and Spirakis, P. G. (2006). Weighted Random Sampling with a
Reservoir. *Information Processing Letters* 97(5), 181-185.
