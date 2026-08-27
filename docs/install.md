# Install

```bash
pip install lexicase
```

That gives you NumPy and every selection method. The optional backends are extras:

```bash
pip install "lexicase[jax]"        # adds the JAX backend
pip install "lexicase[torch]"      # adds the Torch backend
pip install "lexicase[jax,torch]"  # both
```

Python 3.9 or newer. NumPy 1.20 or newer.

## Importing never pulls in a backend

`import lexicase` does not import jax or torch, even when they are installed. A
JAX array or a Torch tensor cannot exist unless you have already imported that
library yourself, so the package detects one by looking in `sys.modules` rather
than by importing anything.

This was [issue #1](https://github.com/ryanboldi/lexicase/issues/1), and there is
a subprocess test that fails the build if it ever comes back.

```python
import sys
import lexicase
assert "jax" not in sys.modules
assert "torch" not in sys.modules
```

If you ask for a backend you do not have installed, you get an error naming the
extra:

```python
lexicase.lexicase_selection(fitness, 10, backend="jax")
# ImportError: The JAX backend requires jax. Install it with: pip install 'lexicase[jax]'
```

## Development install

```bash
git clone https://github.com/ryanboldi/lexicase
cd lexicase
pip install -e ".[dev,jax,torch,examples,docs]"
pytest tests/
```

The `dev` extra brings pytest, coverage, ruff, and Hypothesis. `examples` brings
matplotlib, which only `examples/07_diversity_over_generations.py` needs. `docs`
brings mkdocs-material and mkdocstrings.
