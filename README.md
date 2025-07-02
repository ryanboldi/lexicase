# 🧬 Lexicase Selection Library

A fast, vectorized lexicase selection implementation supporting both NumPy and JAX backends.

## 🎯 What it does

Lexicase selection is a parent selection method used in evolutionary computation that evaluates individuals on test cases in random order, keeping only those that perform best on each case. This library provides efficient implementations of several lexicase variants:

- **Base Lexicase**: Standard lexicase selection algorithm
- **Epsilon Lexicase**: Allows individuals within epsilon of the best to be considered equally good (uses adaptive MAD-based epsilon by default)
- **Downsampled Lexicase**: Uses random subsets of test cases to increase diversity

## 📦 Installation

### PyPI
```bash
pip install lexicase
```

To use the numpy backend, you can use the following command:

```bash
pip install lexicase[numpy]
```

To use the JAX backend, you need to install JAX and JAXlib and then use the following command:

```bash
pip install lexicase[jax]
```

## Source Installation

To install from source, clone the repository and then run the following command:

```bash
pip install -e .
```

### Development Installation
```bash
pip install .[dev]  # Includes pytest and coverage tools
```

## 🚀 Quick Start

```python
import numpy as np
import lexicase

# Create a fitness matrix (individuals × test cases)
# Higher values = better performance
fitness_matrix = np.array([
    [10, 5, 8],  # Individual 0
    [8, 9, 6],   # Individual 1  
    [6, 7, 9],   # Individual 2
    [4, 3, 7]    # Individual 3
])

# Select 5 individuals using standard lexicase
selected = lexicase.lexicase_selection(
    fitness_matrix, 
    num_selected=5, 
    seed=42
)
print(f"Selected individuals: {selected}")

# Use epsilon lexicase with adaptive MAD-based epsilon (recommended)
selected_eps = lexicase.epsilon_lexicase_selection(
    fitness_matrix, 
    num_selected=5, 
    seed=42
)
print(f"Epsilon lexicase selected: {selected_eps}")
```

## 🔧 Backend Selection

Switch between NumPy and JAX backends:

```python
import lexicase

# Use NumPy backend (default)
lexicase.set_backend("numpy")

# Use JAX backend for GPU acceleration
lexicase.set_backend("jax")

# Check current backend
print(f"Current backend: {lexicase.get_backend()}")
```

## 📊 All Selection Methods

### Standard Lexicase
```python
selected = lexicase.lexicase_selection(fitness_matrix, num_selected=10, seed=42)
```

### Epsilon Lexicase
```python
# Recommended: Use adaptive MAD-based epsilon (automatic)
selected = lexicase.epsilon_lexicase_selection(
    fitness_matrix, 
    num_selected=10, 
    seed=42
)

# Alternative: Manual epsilon specification
selected = lexicase.epsilon_lexicase_selection(
    fitness_matrix, 
    num_selected=10, 
    epsilon=0.5,  # Tolerance for "equal" performance
    seed=42
)
```

### Downsampled Lexicase
```python
selected = lexicase.downsample_lexicase_selection(
    fitness_matrix,
    num_selected=10,
    downsample_size=5,  # Use only 5 random test cases per selection
    seed=42
)
```


## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=lexicase --cov-report=html
```

## 🔬 Algorithm Details

**Lexicase Selection Process:**
1. Shuffle the order of test cases
2. Start with all individuals as candidates
3. For each test case (in shuffled order):
   - Find the best performance on this case
   - Keep only individuals matching the best performance
   - If only one individual remains, select it
4. If multiple individuals remain after all cases, select randomly

**Epsilon Lexicase:** Considers individuals within `epsilon` of the best performance as equally good. By default, uses adaptive epsilon values based on the Median Absolute Deviation (MAD) of fitness values for each test case, providing robust and data-driven tolerance levels.

**Downsampled Lexicase:** Uses only a random subset of test cases, increasing selection diversity.

## 📈 Performance Tips

- Use JAX backend for large matrices and GPU acceleration
- Downsampled variants are faster and often more diverse
- For epsilon lexicase, the adaptive MAD-based epsilon (default) is recommended for most use cases
- Use seeds for reproducible results

## 📚 Citation

If you use this library in your research, please cite:

```bibtex
@software{lexicase_selection,
  title={Lexicase Selection Library},
  author={Ryan Bahlous-Boldi},
  year={2024},
  url={https://github.com/ryanboldi/lexicase}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## TODOs:

- [ ] make jax implementaion faster, and jittable.
- [ ] Add informed down-sampling
- [ ] Add some demo notebooks

## 🔗 References

- Spector, L. (2012). Assessment of Problem Modality by Differential Performance of Lexicase Selection in Genetic Programming: A Preliminary Report. In Companion Publication of the 2012 Genetic and Evolutionary Computation Conference, GECCO’12 Companion. ACM Press. pp. 401 - 408.
- Helmuth, T., L. Spector, and J. Matheson. (2014). Solving Uncompromising Problems with Lexicase Selection. In IEEE Transactions on Evolutionary Computation, vol. 19, no. 5, pp. 630 - 643.
- La Cava, W., L. Spector, and K. Danai (2016). Epsilon-lexicase selection for regression. GECCO '16: Proceedings of the Genetic and Evolutionary Computation Conference, pp. 741 - 748.
- Hernandez, J. G., A. Lalejini, E. Dolson, and C. Ofria (2019). Random subsampling improves performance in lexicase selection. GECCO '19: Proceedings of the Genetic and Evolutionary Computation Conference Companion, pp. 2028 - 2031
