# Lexicase selection

Take the cases in a random order. Start with the whole population as candidates.
For each case in turn, keep only the candidates that are best on it. Stop when one
candidate is left, or when the cases run out, in which case pick uniformly from
whoever is still standing. Redraw the case order for every selection event.

!!! quote "Paper"
    Helmuth, T., Spector, L., and Matheson, J. (2015). Solving Uncompromising
    Problems with Lexicase Selection. *IEEE Transactions on Evolutionary
    Computation* 19(5), 630-643.

    Earlier: Spector, L. (2012). Assessment of Problem Modality by Differential
    Performance of Lexicase Selection in Genetic Programming. GECCO '12
    Companion, pp. 401-408.

## When to use it

Use it when your cases are discrete, so exact ties happen: pass/fail tests,
integer counts, boolean outputs. Program synthesis is the canonical case.

Do not use it directly on continuous errors. With floats, no two individuals ever
tie exactly, so the first case drawn decides the whole event and the method
collapses into "pick the best on one random case". Use
[epsilon lexicase](epsilon.md) instead.

## What it guarantees

Any individual it returns is best in the population on at least one case. The
converse is not true: being elite on a case is necessary but not sufficient, since
another individual can tie you there and then beat you later in the order. An
individual that is dominated by another, meaning weakly worse on every case, can
never be selected.

## Usage

```python
from lexicase import lexicase_selection

selected = lexicase_selection(fitness, num_selected=100, seed=0)
selected = lexicase_selection(fitness, num_selected=100, seed=0, elitism=2)
```

::: lexicase.lexicase_selection
