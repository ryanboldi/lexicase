"""Guards for issue #1: importing lexicase must never import jax."""

import subprocess
import sys
import textwrap

import numpy as np


def run_python(code):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )


def test_importing_lexicase_does_not_import_jax():
    result = run_python(
        """
        import sys
        import lexicase
        from lexicase import epsilon_lexicase_selection, lexicase_selection
        leaked = sorted(m for m in sys.modules if m == "jax" or m.startswith("jax."))
        assert not leaked, leaked
        print("clean")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_submodules_do_not_import_jax():
    result = run_python(
        """
        import sys
        import lexicase.backends, lexicase.dispatch, lexicase.numpy_impl, lexicase.utils
        leaked = sorted(m for m in sys.modules if m == "jax" or m.startswith("jax."))
        assert not leaked, leaked
        print("clean")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_numpy_path_works_when_jax_is_not_installed():
    result = run_python(
        """
        import sys

        class BlockJax:
            def find_module(self, name, path=None):
                return self.find_spec(name, path)

            def find_spec(self, name, path=None, target=None):
                if name == "jax" or name.startswith("jax."):
                    raise ImportError("jax is not installed")
                return None

        sys.meta_path.insert(0, BlockJax())

        import numpy as np
        from lexicase import lexicase_selection, epsilon_lexicase_selection

        fitness = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
        assert len(lexicase_selection(fitness, 5, seed=0)) == 5
        assert len(epsilon_lexicase_selection(fitness, 5, seed=0)) == 5

        try:
            lexicase_selection(fitness, 5, seed=0, backend="jax")
        except ImportError as exc:
            assert "lexicase[jax]" in str(exc), str(exc)
        else:
            raise AssertionError("expected ImportError")
        print("clean")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_is_jax_array_is_false_without_jax_imported():
    from lexicase.backends import is_jax_array

    assert not is_jax_array(np.zeros((2, 2)))
    assert not is_jax_array([[1, 2], [3, 4]])
