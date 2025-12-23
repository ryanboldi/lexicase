# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2024-12-19

### Changed
- **Breaking**: Removed JAX backend support. The library now uses NumPy exclusively.
- Simplified architecture with single NumPy implementation
- NumPy is now a required dependency (previously optional)
- Streamlined public API

### Added
- Comprehensive test suite (208 tests covering algorithm correctness, edge cases, and stress tests)
- Informed downsampled lexicase selection
- `getting_started.ipynb` example notebook

### Removed
- JAX implementation and JAX-specific code
- Automatic dispatch system (no longer needed with single backend)
- JAX optional dependency
- Python script examples (replaced with Jupyter notebooks)

## [0.2.0] - 2024-12-18

### Added
- Elitism parameter for all selection methods
- Automatic array-type dispatch between NumPy and JAX backends
- MAD-based adaptive epsilon for epsilon lexicase (now the default)

### Changed
- Epsilon lexicase now uses MAD-based epsilon by default when epsilon is not specified

## [0.1.0] - 2024-12-17

### Added
- Initial release
- Standard lexicase selection
- Epsilon lexicase selection
- Downsampled lexicase selection
- NumPy and JAX implementations
