# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Release dates are the PyPI upload dates for each version. Earlier revisions of
this file dated the first three releases to December 2024, which was wrong: the
first commit landed on 2025-06-14 and `lexicase` 0.1.0 went to PyPI on
2025-06-30.

## [0.4.0] - unreleased

### Added
- JAX backend, restored for every public function and rewritten. Kernels use
  static shapes, `lax.scan` over cases, and `vmap` over selection events, so
  they are jittable and vmappable. See the backends table in the README.
- `backend=` on every selection function, taking "auto", "numpy", or "jax".
  "auto" follows the input array type.
- `jax` optional dependency group: `pip install lexicase[jax]`.
- Epsilon lexicase modes `static`, `semi-dynamic` (the default, unchanged
  behaviour), and `dynamic`, following La Cava et al. (2019) Algorithms 2 to 4.
- `batch_lexicase_selection`, following Aenugu and Spector (2019).
- `cohort_lexicase_selection`, following Hernandez et al. (2019).
- `plexicase_selection` and `plexicase_probabilities`, following
  Ding et al. (2023).
- `dalex_selection`, following Ni et al. (2024).
- `case_weights` on `lexicase_selection` and `epsilon_lexicase_selection` for
  non-uniform case ordering.
- `py.typed`, so type checkers see the annotations.
- GitHub Actions CI: pytest on Python 3.9 to 3.13, numpy-only and numpy+jax,
  plus a ruff lint job.
- `examples/` with runnable scripts, `benchmarks/bench.py`,
  `benchmarks/bench_jax_vs_numpy.py` and the chart it draws, `CONTRIBUTING.md`,
  `CITATION.cff`, and issue templates.

### Fixed
- Issue #1: importing `lexicase` no longer imports jax, and never did import it
  in 0.3.0 only because the JAX backend had been deleted. Backend detection now
  reads `sys.modules`, so the import stays clean even when jax is installed. A
  subprocess test enforces this.
- Case distances in informed downsampling are computed by matrix multiplication
  instead of a Python double loop. Same values, much faster on large case sets.
- `baselines/baselines.py` had a broken `fitness_proportionate_selection` that
  passed a 2-D probability array to `np.random.choice`, and a
  `tournament_selection` that took `argmax` over a 2-D slice. Both are replaced
  by seeded implementations in `baselines/`.

### Changed
- `requires-python` raised to >=3.9.
- Keywords and classifiers updated.
- Workflow tokens are scoped per job. `pages: write` and `id-token: write` now
  sit on the docs deploy job instead of the whole workflow, so the build job that
  runs PR code holds a read-only token. CI declares `contents: read` explicitly
  rather than inheriting the repository default.
- `benchmarks/bench.py` records the OS and architecture instead of the exact
  kernel build string, so pasting its output somewhere public does not fingerprint
  the machine that ran it.

## [0.3.0] - 2025-12-23

### Changed
- **Breaking**: Removed JAX backend support. The library used NumPy exclusively.
- Simplified architecture with a single NumPy implementation
- NumPy became a required dependency
- Streamlined public API

### Added
- Comprehensive test suite (208 tests covering algorithm correctness, edge
  cases, and stress tests)
- Informed downsampled lexicase selection
- `getting_started.ipynb` example notebook

### Removed
- JAX implementation and JAX-specific code
- Automatic dispatch system
- JAX optional dependency
- Python script examples (replaced with Jupyter notebooks)

## [0.2.0] - 2025-09-11

### Added
- Elitism parameter for all selection methods
- Automatic array-type dispatch between NumPy and JAX backends

## [0.1.1] - 2025-07-02

### Changed
- Epsilon lexicase now uses MAD-based epsilon by default when epsilon is not
  specified

## [0.1.0] - 2025-06-30

### Added
- Initial release
- Standard lexicase selection
- Epsilon lexicase selection
- Downsampled lexicase selection
- NumPy and JAX implementations

[0.4.0]: https://github.com/ryanboldi/lexicase/compare/v0.3.0...v0.4
[0.3.0]: https://pypi.org/project/lexicase/0.3.0/
[0.2.0]: https://pypi.org/project/lexicase/0.2.0/
[0.1.1]: https://pypi.org/project/lexicase/0.1.1/
[0.1.0]: https://pypi.org/project/lexicase/0.1.0/
