# lexicase

Fast, vectorized lexicase selection and its variants, in NumPy, JAX, and PyTorch.

Lexicase selection picks a parent by taking the test cases in a random order and,
one case at a time, throwing out every candidate that is not best on that case,
until one candidate is left. Because the case order is redrawn for every selection
event, an individual that is the best in the population on a single hard case gets
selected as often as a well-rounded individual that is best on none, which is how
lexicase keeps specialists alive where an aggregating method would discard them.
That property is why it is the default parent selection method for program
synthesis, and why it does well on any problem where every case has to be passed
at once.

```python
import numpy as np
from lexicase import lexicase_selection

fitness = np.array([
    [10, 5, 8],
    [8, 9, 6],
    [6, 7, 9],
    [4, 3, 7],
])

lexicase_selection(fitness, num_selected=5, seed=42)
# array([2, 0, 1, 2, 2])
```

## What is here

- Eight selection methods, each implemented from a published algorithm and citing
  it: [lexicase](variants/lexicase.md), [epsilon lexicase](variants/epsilon.md)
  in all three modes, [downsampled](variants/downsample.md),
  [informed downsampled](variants/informed-downsample.md),
  [batch](variants/batch.md), [cohort](variants/cohort.md),
  [plexicase](variants/plexicase.md), and [DALex](variants/dalex.md).
- Three [backends](backends.md). The backend follows the type of array you pass
  in. Importing the package never imports jax or torch.
- [Benchmarks](benchmarks.md) produced by a script in the repository, on a stated
  machine with stated library versions.
- Nine runnable examples in `examples/`.

## Install

```bash
pip install lexicase
```

See [Install](install.md) for the optional backends.

## Citation

```bibtex
@article{boldi2024informed,
  title = {Informed Down-Sampled Lexicase Selection: Identifying Productive
           Training Cases for Efficient Problem Solving},
  author = {Boldi, Ryan and Briesch, Martin and Sobania, Dominik and Lalejini,
            Alexander and Helmuth, Thomas and Rothlauf, Franz and Ofria, Charles
            and Spector, Lee},
  journal = {Evolutionary Computation},
  volume = {32},
  number = {4},
  pages = {307--337},
  year = {2024},
  doi = {10.1162/evco_a_00346},
}
```
