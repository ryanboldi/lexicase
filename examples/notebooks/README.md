# Notebooks

Older walkthroughs, kept because they cover the same ground more slowly and with
more prose than the scripts in the parent directory.

| Notebook | Needs |
|---|---|
| `getting_started.ipynb` | numpy |
| `lexicase_demo.ipynb` | numpy, and `baselines/` from the repository root |
| `simple_ea_demo.ipynb` | numpy |
| `performance_comparison.ipynb` | numpy, jax |

Run them from this directory. `lexicase_demo.ipynb` puts the repository root on
`sys.path` so it can import `baselines`.

Outputs are stripped before committing. `lexicase_demo.ipynb` used to carry about
1 MB of stored output.
