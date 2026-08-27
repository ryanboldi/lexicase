"""
Torch backend: device preservation, distributional agreement, and no host syncs.

The no-sync tests matter because the point of this backend is running lexicase
as a selection operator over a reward tensor that is already on an accelerator.
Anything that pulls a value back to the host stalls the RL training loop.
"""

import itertools
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import lexicase  # noqa: E402
from lexicase import torch_impl  # noqa: E402

TRIALS = 20000

FITNESS = np.array(
    [
        [4.0, 1.0, 2.0],
        [1.0, 4.0, 2.0],
        [2.0, 2.0, 4.0],
        [4.0, 1.0, 1.0],
        [3.0, 3.0, 3.0],
    ]
)

CALLS = {
    "lexicase": lambda f, n, **kw: lexicase.lexicase_selection(f, n, seed=0, **kw),
    "epsilon": lambda f, n, **kw: lexicase.epsilon_lexicase_selection(f, n, seed=0, **kw),
    "downsample": lambda f, n, **kw: lexicase.downsample_lexicase_selection(
        f, n, 2, seed=0, **kw
    ),
    "informed": lambda f, n, **kw: lexicase.informed_downsample_lexicase_selection(
        f, n, 2, seed=0, sample_rate=0.5, threshold=2.5, **kw
    ),
    "batch": lambda f, n, **kw: lexicase.batch_lexicase_selection(f, n, 2, seed=0, **kw),
    "cohort": lambda f, n, **kw: lexicase.cohort_lexicase_selection(f, n, 3, seed=0, **kw),
    "dalex": lambda f, n, **kw: lexicase.dalex_selection(f, n, seed=0, **kw),
}


def exact_lexicase_probabilities(fitness):
    n_individuals, n_cases = fitness.shape
    totals = np.zeros(n_individuals)
    for order in itertools.permutations(range(n_cases)):
        candidates = np.arange(n_individuals)
        for case in order:
            if len(candidates) <= 1:
                break
            values = fitness[candidates, case]
            candidates = candidates[values == values.max()]
        totals[candidates] += 1.0 / len(candidates)
    return totals / math.factorial(n_cases)


def frequencies(indices, n_individuals):
    counts = np.bincount(np.asarray(indices.cpu()), minlength=n_individuals)
    return counts / counts.sum()


def total_variation(a, b):
    return 0.5 * np.abs(a - b).sum()


class TestDispatch:
    @pytest.mark.parametrize("name", sorted(CALLS))
    def test_tensor_in_gives_tensor_out(self, name):
        result = CALLS[name](torch.tensor(FITNESS), 6)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.long
        assert len(result) == 6

    @pytest.mark.parametrize("name", sorted(CALLS))
    def test_indices_are_in_range(self, name):
        result = CALLS[name](torch.tensor(FITNESS), 6)
        assert int(result.min()) >= 0
        assert int(result.max()) < len(FITNESS)

    def test_plexicase_round_trips_through_numpy(self):
        result = lexicase.plexicase_selection(torch.tensor(FITNESS), 6, seed=0)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.long

    def test_backend_override(self):
        forced = lexicase.lexicase_selection(FITNESS, 6, seed=0, backend="torch")
        assert isinstance(forced, torch.Tensor)
        back = lexicase.lexicase_selection(
            torch.tensor(FITNESS), 6, seed=0, backend="numpy"
        )
        assert isinstance(back, np.ndarray)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
    def test_input_dtype_is_accepted(self, dtype):
        matrix = torch.tensor(FITNESS).to(dtype)
        result = lexicase.lexicase_selection(matrix, 6, seed=0)
        assert result.dtype == torch.long

    def test_torch_is_available_reports_true_here(self):
        assert lexicase.torch_is_available()


class TestDistribution:
    def test_matches_exact_lexicase_probabilities(self):
        expected = exact_lexicase_probabilities(FITNESS)
        observed = frequencies(
            lexicase.lexicase_selection(torch.tensor(FITNESS), TRIALS, seed=7),
            len(FITNESS),
        )
        assert total_variation(expected, observed) < 0.02

    def test_numpy_and_torch_lexicase_are_distributionally_equivalent(self):
        numpy_frequencies = np.bincount(
            lexicase.lexicase_selection(FITNESS, TRIALS, seed=11), minlength=len(FITNESS)
        ) / TRIALS
        torch_frequencies = frequencies(
            lexicase.lexicase_selection(torch.tensor(FITNESS), TRIALS, seed=11),
            len(FITNESS),
        )
        assert total_variation(numpy_frequencies, torch_frequencies) < 0.02

    @pytest.mark.parametrize("mode", ["static", "semi-dynamic", "dynamic"])
    def test_epsilon_modes_agree_with_numpy(self, mode):
        kwargs = {} if mode == "dynamic" else {"epsilon": 1.0}
        numpy_frequencies = np.bincount(
            lexicase.epsilon_lexicase_selection(
                FITNESS, TRIALS, seed=5, mode=mode, **kwargs
            ),
            minlength=len(FITNESS),
        ) / TRIALS
        torch_frequencies = frequencies(
            lexicase.epsilon_lexicase_selection(
                torch.tensor(FITNESS), TRIALS, seed=5, mode=mode, **kwargs
            ),
            len(FITNESS),
        )
        assert total_variation(numpy_frequencies, torch_frequencies) < 0.03

    def test_downsample_agrees_with_numpy(self):
        numpy_frequencies = np.bincount(
            lexicase.downsample_lexicase_selection(FITNESS, TRIALS, 2, seed=13),
            minlength=len(FITNESS),
        ) / TRIALS
        torch_frequencies = frequencies(
            lexicase.downsample_lexicase_selection(
                torch.tensor(FITNESS), TRIALS, 2, seed=13
            ),
            len(FITNESS),
        )
        assert total_variation(numpy_frequencies, torch_frequencies) < 0.02

    def test_dalex_agrees_with_numpy(self):
        numpy_frequencies = np.bincount(
            lexicase.dalex_selection(
                FITNESS, TRIALS, seed=19, particularity_pressure=5.0
            ),
            minlength=len(FITNESS),
        ) / TRIALS
        torch_frequencies = frequencies(
            lexicase.dalex_selection(
                torch.tensor(FITNESS), TRIALS, seed=19, particularity_pressure=5.0
            ),
            len(FITNESS),
        )
        assert total_variation(numpy_frequencies, torch_frequencies) < 0.02

    def test_same_seed_is_reproducible(self):
        matrix = torch.tensor(FITNESS)
        first = lexicase.lexicase_selection(matrix, 50, seed=99)
        second = lexicase.lexicase_selection(matrix, 50, seed=99)
        assert torch.equal(first, second)


class TestNoHostSync:
    """Every call to a torch kernel must run without pulling a value to the host."""

    @pytest.fixture
    def no_sync(self, monkeypatch):
        def forbidden(name):
            def raiser(*args, **kwargs):
                raise AssertionError(f"torch kernel synchronized with the host via {name}")

            return raiser

        for name in ("item", "tolist", "numpy", "cpu", "__bool__", "__int__", "__float__"):
            monkeypatch.setattr(torch.Tensor, name, forbidden(name), raising=False)

    @pytest.mark.parametrize(
        "call",
        [
            lambda f: torch_impl.torch_lexicase_selection(f, 8, seed=0),
            lambda f: torch_impl.torch_lexicase_selection(f, 8, seed=0, elitism=2),
            lambda f: torch_impl.torch_epsilon_lexicase_selection(f, 8, 0.5, seed=0),
            lambda f: torch_impl.torch_epsilon_lexicase_selection(
                f, 8, 0.5, seed=0, mode="static"
            ),
            lambda f: torch_impl.torch_epsilon_lexicase_selection(
                f, 8, 0.0, seed=0, mode="dynamic"
            ),
            lambda f: torch_impl.torch_downsample_lexicase_selection(f, 8, 2, seed=0),
            lambda f: torch_impl.torch_informed_downsample_lexicase_selection(
                f, 8, 2, seed=0, sample_rate=0.5
            ),
            lambda f: torch_impl.torch_batch_lexicase_selection(f, 8, 2, seed=0),
            lambda f: torch_impl.torch_batch_lexicase_selection(
                f, 8, 2, seed=0, threshold=2.0
            ),
            lambda f: torch_impl.torch_cohort_lexicase_selection(f, 8, 3, seed=0),
            lambda f: torch_impl.torch_dalex_selection(f, 8, seed=0),
            lambda f: torch_impl.torch_compute_mad_epsilon(f),
        ],
        ids=[
            "lexicase",
            "lexicase-elitism",
            "epsilon-semi-dynamic",
            "epsilon-static",
            "epsilon-dynamic",
            "downsample",
            "informed",
            "batch",
            "batch-threshold",
            "cohort",
            "dalex",
            "mad",
        ],
    )
    def test_kernel_does_not_sync(self, no_sync, call):
        result = call(torch.tensor(FITNESS))
        assert result.shape[0] > 0

    def test_the_guard_itself_catches_a_sync(self, no_sync):
        """Negative control, so a broken monkeypatch cannot make the suite pass."""
        with pytest.raises(AssertionError, match="synchronized with the host"):
            torch.tensor(FITNESS).sum().item()


class TestInformedDownsampleThreshold:
    def test_missing_threshold_is_refused_rather_than_guessed(self):
        with pytest.raises(ValueError, match="will not infer one"):
            lexicase.informed_downsample_lexicase_selection(
                torch.tensor(FITNESS), 6, 2, seed=0, sample_rate=0.5
            )

    def test_explicit_threshold_works(self):
        selected = lexicase.informed_downsample_lexicase_selection(
            torch.tensor(FITNESS), 6, 2, seed=0, sample_rate=0.5, threshold=2.5
        )
        assert isinstance(selected, torch.Tensor)
        assert len(selected) == 6


class TestNaNPolicy:
    def test_nan_is_the_worst_value_on_its_case(self):
        matrix = torch.tensor([[float("nan"), 5.0], [1.0, 1.0]])
        selected = lexicase.lexicase_selection(matrix, 200, seed=0)
        counts = np.bincount(np.asarray(selected.cpu()), minlength=2)
        # Individual 0 loses case 0 outright and wins case 1, so it gets about half.
        assert counts[0] > 0 and counts[1] > 0

    def test_all_nan_individual_is_never_selected(self):
        matrix = torch.tensor([[1.0, 1.0], [2.0, 2.0], [float("nan"), float("nan")]])
        selected = lexicase.lexicase_selection(matrix, 200, seed=0)
        assert 2 not in set(np.asarray(selected.cpu()).tolist())

    def test_backends_agree_on_nan_handling(self):
        matrix = np.array([[3.0, np.nan, 2.0], [1.0, 3.0, 2.0], [np.nan, 1.0, 4.0]])
        numpy_frequencies = np.bincount(
            lexicase.lexicase_selection(matrix, 4000, seed=2), minlength=3
        ) / 4000
        torch_frequencies = frequencies(
            lexicase.lexicase_selection(torch.tensor(matrix), 4000, seed=2), 3
        )
        assert total_variation(numpy_frequencies, torch_frequencies) < 0.03
