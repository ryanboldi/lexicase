# Lexicase Selection Library

A fast, vectorized lexicase selection implementation using NumPy for evolutionary computation.

## What it does

Lexicase selection is a parent selection method used in evolutionary computation that evaluates individuals on test cases in random order, keeping only those that perform best on each case. This library provides efficient implementations of several lexicase variants:

- **Base Lexicase**: Standard lexicase selection algorithm
- **Epsilon Lexicase**: Allows individuals within epsilon of the best to be considered equally good (uses adaptive MAD-based epsilon by default)
- **Downsampled Lexicase**: Uses random subsets of test cases for faster selection
- **Informed Downsampled Lexicase**: Intelligently selects informative test cases
- **Elitism**: Optionally reserve slots for top individuals before running lexicase

## Installation

```bash
pip install lexicase
```

### Development Installation

```bash
pip install -e .[dev]  # Includes pytest and coverage tools
```

## Quick Start

```python
import numpy as np
from lexicase import lexicase_selection

# Create a fitness matrix (individuals x test cases)
# Higher values = better performance
fitness_matrix = np.array([
    [10, 5, 8],  # Individual 0
    [8, 9, 6],   # Individual 1
    [6, 7, 9],   # Individual 2
    [4, 3, 7]    # Individual 3
])

# Select 5 individuals using standard lexicase
selected = lexicase_selection(fitness_matrix, num_selected=5, seed=42)
print(f"Selected individuals: {selected}")
# Output: Selected individuals: [1 0 2 1 0]
```

## Selection Methods

### Standard Lexicase

```python
from lexicase import lexicase_selection

selected = lexicase_selection(fitness_matrix, num_selected=10, seed=42)
```

### Epsilon Lexicase

Uses adaptive MAD-based epsilon by default for robust tolerance levels:

```python
from lexicase import epsilon_lexicase_selection

# Recommended: Use adaptive MAD-based epsilon (automatic)
selected = epsilon_lexicase_selection(fitness_matrix, num_selected=10, seed=42)

# Alternative: Manual epsilon specification
selected = epsilon_lexicase_selection(
    fitness_matrix,
    num_selected=10,
    epsilon=0.5,  # Tolerance for "equal" performance
    seed=42
)
```

### Downsampled Lexicase

Faster selection using random subsets of test cases:

```python
from lexicase import downsample_lexicase_selection

selected = downsample_lexicase_selection(
    fitness_matrix,
    num_selected=10,
    downsample_size=5,  # Use only 5 random test cases per selection
    seed=42
)
```

### Informed Downsampled Lexicase

Intelligently selects informative test cases based on population statistics:

```python
from lexicase import informed_downsample_lexicase_selection

selected = informed_downsample_lexicase_selection(
    fitness_matrix,
    num_selected=10,
    downsample_size=5,
    seed=42,
    sample_rate=0.01,  # Fraction of population to sample for distance calculation
)
```

### Elitism

All methods support elitism to preserve top performers:

```python
from lexicase import lexicase_selection, epsilon_lexicase_selection

# Standard lexicase with 2 elites
selected = lexicase_selection(
    fitness_matrix,
    num_selected=10,
    seed=42,
    elitism=2,  # Top 2 by total fitness always included
)

# Epsilon lexicase with 1 elite
selected_eps = epsilon_lexicase_selection(
    fitness_matrix,
    num_selected=10,
    seed=42,
    elitism=1,
)
```

Notes:
- Elites are determined by total fitness (sum across cases)
- `elitism` must be between 0 and `num_selected` and cannot exceed the number of individuals

## Algorithm Details

**Lexicase Selection Process:**
1. Shuffle the order of test cases
2. Start with all individuals as candidates
3. For each test case (in shuffled order):
   - Find the best performance on this case
   - Keep only individuals matching the best performance
   - If only one individual remains, select it
4. If multiple individuals remain after all cases, select randomly

**Epsilon Lexicase:** Considers individuals within `epsilon` of the best performance as equally good. By default, uses adaptive epsilon values based on the Median Absolute Deviation (MAD) of fitness values for each test case.

**Downsampled Lexicase:** Uses only a random subset of test cases, increasing selection diversity and reducing computation time.

**Informed Downsampled Lexicase:** Selects test cases that are maximally different from each other based on population performance patterns.

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=lexicase --cov-report=html
```

## API Reference

All selection functions share a common signature:

| Parameter | Type | Description |
|-----------|------|-------------|
| `fitness_matrix` | array-like | Shape (n_individuals, n_cases). Higher = better. |
| `num_selected` | int | Number of individuals to select |
| `seed` | int, optional | Random seed for reproducibility |
| `elitism` | int, optional | Number of top individuals to always include (default: 0) |

Additional parameters:
- `epsilon_lexicase_selection`: `epsilon` (float or array, optional) - tolerance threshold
- `downsample_lexicase_selection`: `downsample_size` (int) - number of cases to sample
- `informed_downsample_lexicase_selection`: `downsample_size`, `sample_rate`, `threshold`

## Citation

If you use this library in your research, please cite:

```bibtex
@software{lexicase_selection,
  title={Lexicase Selection Library},
  author={Ryan Bahlous-Boldi},
  year={2024},
  url={https://github.com/ryanboldi/lexicase}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

- Spector, L. (2012). Assessment of Problem Modality by Differential Performance of Lexicase Selection in Genetic Programming: A Preliminary Report. In GECCO'12 Companion. ACM Press. pp. 401-408.
- Helmuth, T., L. Spector, and J. Matheson. (2014). Solving Uncompromising Problems with Lexicase Selection. IEEE Transactions on Evolutionary Computation, vol. 19, no. 5, pp. 630-643.
- La Cava, W., L. Spector, and K. Danai (2016). Epsilon-lexicase selection for regression. GECCO '16, pp. 741-748.
- Hernandez, J. G., A. Lalejini, E. Dolson, and C. Ofria (2019). Random subsampling improves performance in lexicase selection. GECCO '19 Companion, pp. 2028-2031.
