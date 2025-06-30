# 🧬 Lexicase Selection Library

A fast, vectorized lexicase selection implementation supporting both NumPy and JAX backends.

## 🎯 What it does

Lexicase selection is a parent selection method used in evolutionary computation that evaluates individuals on test cases in random order, keeping only those that perform best on each case. This library provides efficient implementations of several lexicase variants:

- **Base Lexicase**: Standard lexicase selection algorithm
- **Epsilon Lexicase**: Allows individuals within epsilon of the best to be considered equally good
- **Downsampled Lexicase**: Uses random subsets of test cases to increase diversity

## TODOs:

- [ ] Add some demo notebooks
- [ ] Add informed down-sampling
- [ ] Add MAD calculation for automatic epsilon selection

## 📦 Installation

### NumPy Backend
```bash
pip install .[numpy]
```

### JAX Backend  
```bash
pip install .[jax]
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

# Use epsilon lexicase for more diversity
selected_eps = lexicase.epsilon_lexicase_selection(
    fitness_matrix, 
    num_selected=5, 
    epsilon=1.0,
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

**Epsilon Lexicase:** Considers individuals within `epsilon` of the best performance as equally good.

**Downsampled Lexicase:** Uses only a random subset of test cases, increasing selection diversity.

## 📈 Performance Tips

- Use JAX backend for large matrices and GPU acceleration
- Downsampled variants are faster and often more diverse
- Set appropriate epsilon values (typically 0.1-1.0 of fitness range)
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

## 🔗 References

- Spector, L. (2012). Assessment of problem modality by differential performance of lexicase selection in genetic programming. GECCO.
- La Cava, W., et al. (2019). Epsilon-lexicase selection for regression. GECCO.
- Hernandez, J. G., et al. (2019). Random subsampling improves performance in lexicase selection. GECCO. 
