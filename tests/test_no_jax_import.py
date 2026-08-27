"""Guards for issue #1: importing lexicase must never import an optional backend."""

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


LEAK_CHECK = """
        leaked = sorted(
            m for m in sys.modules
            if m in ("jax", "torch") or m.startswith("jax.") or m.startswith("torch.")
        )
        assert not leaked, leaked
        print("clean")
        """


def test_importing_lexicase_does_not_import_optional_backends():
    result = run_python(
        """
        import sys
        import lexicase
        from lexicase import epsilon_lexicase_selection, lexicase_selection
        """
        + LEAK_CHECK
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_submodules_do_not_import_optional_backends():
    result = run_python(
        """
        import sys
        import lexicase.backends, lexicase.dispatch, lexicase.numpy_impl, lexicase.utils
        """
        + LEAK_CHECK
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_numpy_path_works_when_torch_is_not_installed():
    result = run_python(
        """
        import sys

        class BlockTorch:
            def find_spec(self, name, path=None, target=None):
                if name == "torch" or name.startswith("torch."):
                    raise ImportError("torch is not installed")
                return None

        sys.meta_path.insert(0, BlockTorch())

        import numpy as np
        from lexicase import lexicase_selection

        fitness = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
        assert len(lexicase_selection(fitness, 5, seed=0)) == 5

        try:
            lexicase_selection(fitness, 5, seed=0, backend="torch")
        except ImportError as exc:
            assert "lexicase[torch]" in str(exc), str(exc)
        else:
            raise AssertionError("expected ImportError")
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


def test_array_detection_is_false_for_plain_numpy():
    from lexicase.backends import is_jax_array, is_torch_tensor

    assert not is_jax_array(np.zeros((2, 2)))
    assert not is_jax_array([[1, 2], [3, 4]])
    assert not is_torch_tensor(np.zeros((2, 2)))
    assert not is_torch_tensor([[1, 2], [3, 4]])
