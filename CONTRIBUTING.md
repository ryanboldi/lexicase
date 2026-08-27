# Contributing

Bug reports, new variants, and speedups are all welcome.

## Setup

```bash
git clone https://github.com/ryanboldi/lexicase
cd lexicase
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,jax]"
```

The `jax` extra is optional. Without it the JAX tests skip and everything else
still runs, which is exactly the configuration one of the CI jobs checks.

## Before you open a PR

```bash
pytest tests/
ruff check lexicase/ tests/ baselines/ examples/ benchmarks/
```

CI runs pytest on Python 3.9 to 3.13, once with numpy alone and once with numpy
plus jax, and runs ruff. All three have to pass.

## Adding a selection variant

Every variant in this package is implemented from a published algorithm, and
the docstring cites the paper, the section, and the algorithm or equation
number it follows. If the paper's pseudocode and its prose disagree, say so in
the docstring and expose both readings as a parameter rather than picking one
silently. `batch_lexicase_selection` is the worked example of that.

What a new variant needs:

1. A NumPy kernel in `lexicase/numpy_impl.py`.
2. A public wrapper in `lexicase/dispatch.py` that validates inputs and
   dispatches. Keep `elitism` and `backend` for consistency with the rest.
3. A JAX kernel in `lexicase/jax_impl.py`, if the algorithm works with static
   shapes. If it does not, say so in the wrapper's docstring and in the backend
   table in the README, and fall back to the NumPy kernel.
4. Tests. Not smoke tests. For anything small enough, compare the empirical
   selection frequencies against exact probabilities from enumerating every
   case ordering, the way `tests/test_variants.py` does. For anything with two
   backends, add a distributional equivalence test to
   `tests/test_cross_backend.py`.
5. An export in `lexicase/__init__.py`, a row in the README's variant table,
   and a CHANGELOG entry.

## Things that will get a PR sent back

- Importing jax at module load. Nothing outside `lexicase/jax_impl.py` may
  import jax, and `tests/test_no_jax_import.py` enforces it.
- Changing the output of a seeded call without saying so. Seeded results are
  part of the API. If a change moves them, it goes in the CHANGELOG and the
  affected tests and README examples get updated in the same commit.
- Unseeded randomness anywhere in tests, examples, or benchmarks.
- Performance claims without a script in `benchmarks/` that produces them.

## Docs

The site under `docs/` is mkdocs-material, published to GitHub Pages by
`.github/workflows/docs.yml` on every push to main.

```bash
pip install -e ".[docs,jax,torch]"
mkdocs serve
```

`mkdocs build --strict` has to pass, which means every public function needs a
docstring whose parameters are all annotated. A new variant needs a page under
`docs/variants/` with its paper citation and guidance on when to reach for it,
plus a row in `docs/variants/index.md` and an entry in the `nav` in `mkdocs.yml`.

## Style

Flat and legible over clever. Docstrings on the public API, few comments
elsewhere. No em dashes.

## Backends

`lexicase.backends` decides which kernel runs. It detects a JAX array by
looking up `jax` in `sys.modules`, on the reasoning that a JAX array cannot
exist unless the caller already imported jax. That is deliberate, and it is
what makes `import lexicase` free of a jax import even when jax is installed.
