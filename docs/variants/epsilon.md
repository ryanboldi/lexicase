# Epsilon lexicase selection

Same filtering as [standard lexicase](lexicase.md), except that a candidate
survives a case if it is within `epsilon` of the best rather than exactly equal to
it. This is what makes lexicase work on continuous errors.

!!! quote "Paper"
    La Cava, W., Helmuth, T., Spector, L., and Moore, J. H. (2019). A Probabilistic
    and Multi-Objective Analysis of Lexicase Selection and Epsilon-Lexicase
    Selection. *Evolutionary Computation* 27(3), 377-402. Algorithms 2, 3, and 4.

    Originally: La Cava, W., Spector, L., and Danai, K. (2016). Epsilon-Lexicase
    Selection for Regression. GECCO '16, pp. 741-748.

## The three modes

The modes differ in where the elite value and the tolerance come from.

| `mode` | Elite from | Epsilon from | Notes |
|---|---|---|---|
| `"static"` | whole population | whole population | The fitness matrix is reduced to pass or fail once per call, so any resolution finer than the threshold is discarded |
| `"semi-dynamic"` | current candidate pool | whole population | The default here, and the paper's own recommendation |
| `"dynamic"` | current candidate pool | current candidate pool | Strongest differentiation, and the most expensive, since the tolerance is recomputed at every filtering step |

`"static"` is genuinely different, not just stricter. Reducing the matrix to
pass/fail throws away the information that separates two individuals who both
pass. Two individuals with the same pass pattern become indistinguishable even if
plain lexicase would have separated them.

## The tolerance

With `epsilon=None`, the tolerance for each case is the median absolute deviation
of that case across the population:

$$\varepsilon_t = \text{median}\big(|e_t - \text{median}(e_t)|\big)$$

That is the parameter-free version from the paper, and it is the default. It
adapts on its own: as the population improves on a case, the spread shrinks and so
does the tolerance, which makes an easy case more selective over time.

You can also pass a scalar or one value per case. `mode="dynamic"` recomputes the
tolerance from the candidate pool, so passing an explicit `epsilon` alongside it
is rejected rather than silently ignored.

## Usage

```python
from lexicase import epsilon_lexicase_selection

# MAD tolerance, semi-dynamic. The usual choice.
selected = epsilon_lexicase_selection(-errors, num_selected=100, seed=0)

# explicit per-case tolerance
selected = epsilon_lexicase_selection(-errors, 100, epsilon=0.01, seed=0)

# the other two modes
selected = epsilon_lexicase_selection(-errors, 100, 0.01, seed=0, mode="static")
selected = epsilon_lexicase_selection(-errors, 100, seed=0, mode="dynamic")
```

::: lexicase.epsilon_lexicase_selection
