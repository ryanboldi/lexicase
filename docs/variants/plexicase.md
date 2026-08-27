# Probabilistic lexicase selection (plexicase)

Instead of running a filtering event per parent, build one approximation of the
lexicase selection distribution over the whole population and draw every parent
from it. Once the distribution exists, sampling is free, so the cost stops scaling
with the number of parents you want.

!!! quote "Paper"
    Ding, L., Pantridge, E., and Spector, L. (2023). Probabilistic Lexicase
    Selection. GECCO '23, pp. 1073-1081. Equations 1 to 4.

## How the distribution is built

Exact lexicase selection probabilities are NP-hard to compute, so the paper
approximates them.

Let $E(y_i)$ be the number of cases on which individual $i$ achieves the
population's best fitness. Individuals with $E = 0$, and individuals dominated by
another individual, cannot be selected by lexicase at all and get probability
zero. The rest, the Pareto set boundaries, get

$$h_j(y_i) = \begin{cases} E(y_i) & \text{if } f_j(y_i) = f_j^*(Y) \\ 0 & \text{otherwise}\end{cases}
\qquad
P_j(y_i) = \frac{h_j(y_i)}{\sum_k h_j(y_k)}
\qquad
P(y_i) = \frac{1}{N}\sum_{j=1}^{N} P_j(y_i)$$

and then a temperature:

$$P'(y_i) = \frac{P(y_i)^\alpha}{\sum_k P(y_k)^\alpha}$$

`alpha=1` leaves the distribution alone and is the paper's default. `alpha=0`
makes it uniform over the Pareto set boundaries. Large `alpha` concentrates it on
the most elite individuals.

## Deviations from the paper

- The paper deduplicates identical fitness vectors before comparing. This
  implementation keeps duplicates and uses strict dominance instead, so two
  individuals with identical fitness both survive and get equal probability.
  That is the right generalization for a function that returns indices.
- With `epsilon`, the epsilon-domination relation is made antisymmetric, so two
  individuals that epsilon-dominate each other both survive rather than removing
  each other in an order-dependent way.

## The probabilities on their own

`plexicase_probabilities` returns the distribution without drawing anything, which
is useful for measuring how concentrated selection is on a given population.

```python
from lexicase import plexicase_probabilities

probabilities = plexicase_probabilities(fitness)
effective_population = 1.0 / (probabilities ** 2).sum()
```

## No accelerator kernel

Finding the Pareto set boundaries needs a data-dependent number of candidates,
which cannot be done under `jit` or without a host synchronization. Passing a JAX
array or a Torch tensor runs the NumPy kernel on the host and moves the result
back. This is the one method here that is not safe inside a GPU inner loop.

## Usage

```python
from lexicase import plexicase_selection

selected = plexicase_selection(fitness, 100, seed=0)
selected = plexicase_selection(fitness, 100, seed=0, alpha=2.0)
```

::: lexicase.plexicase_selection

::: lexicase.plexicase_probabilities
