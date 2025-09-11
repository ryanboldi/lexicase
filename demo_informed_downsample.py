#!/usr/bin/env python3
"""
Demonstration of informed downsampled lexicase selection.

This script shows how informed downsampling can select more diverse test cases
compared to random downsampling, potentially leading to better selection outcomes.
"""

import numpy as np
from lexicase import (
    informed_downsample_lexicase_selection, 
    downsample_lexicase_selection,
    lexicase_selection
)

def create_specialist_fitness_matrix():
    """Create a fitness matrix where individuals specialize in different test cases."""
    np.random.seed(42)
    
    # Create 6 individuals and 12 test cases
    # Individuals 0-1: Good at cases 0-3
    # Individuals 2-3: Good at cases 4-7  
    # Individuals 4-5: Good at cases 8-11
    fitness_matrix = np.zeros((6, 12))
    
    # Specialists for first group of cases
    fitness_matrix[0:2, 0:4] = np.random.uniform(8, 10, (2, 4))
    fitness_matrix[0:2, 4:12] = np.random.uniform(1, 3, (2, 8))
    
    # Specialists for second group of cases
    fitness_matrix[2:4, 4:8] = np.random.uniform(8, 10, (2, 4))
    fitness_matrix[2:4, 0:4] = np.random.uniform(1, 3, (2, 4))
    fitness_matrix[2:4, 8:12] = np.random.uniform(1, 3, (2, 4))
    
    # Specialists for third group of cases
    fitness_matrix[4:6, 8:12] = np.random.uniform(8, 10, (2, 4))
    fitness_matrix[4:6, 0:8] = np.random.uniform(1, 3, (2, 8))
    
    return fitness_matrix

def analyze_case_selection_diversity(fitness_matrix, downsample_size=4, n_trials=100):
    """Analyze which cases are selected by different methods."""
    
    informed_case_counts = np.zeros(fitness_matrix.shape[1])
    random_case_counts = np.zeros(fitness_matrix.shape[1])
    
    # Import internal functions to track case selection
    from lexicase.numpy_impl import _compute_case_distances, _farthest_first_traversal
    
    print(f"Running {n_trials} trials to compare case selection diversity...")
    
    for trial in range(n_trials):
        # Informed downsampling
        rng = np.random.default_rng(trial)
        sample_indices = rng.choice(fitness_matrix.shape[0], 
                                  size=max(1, int(fitness_matrix.shape[0] * 0.5)), 
                                  replace=False)
        distances = _compute_case_distances(fitness_matrix, sample_indices)
        informed_cases = _farthest_first_traversal(distances, downsample_size, rng)
        for case in informed_cases:
            informed_case_counts[case] += 1
            
        # Random downsampling
        rng = np.random.default_rng(trial)
        random_cases = rng.choice(fitness_matrix.shape[1], size=downsample_size, replace=False)
        for case in random_cases:
            random_case_counts[case] += 1
    
    return informed_case_counts, random_case_counts

def demonstrate_informed_downsample():
    """Main demonstration of informed downsampled lexicase selection."""
    
    print("=" * 60)
    print("Informed Downsampled Lexicase Selection Demo")
    print("=" * 60)
    
    # Create fitness matrix with specialist individuals
    fitness_matrix = create_specialist_fitness_matrix()
    print(f"Created fitness matrix: {fitness_matrix.shape[0]} individuals, {fitness_matrix.shape[1]} test cases")
    print(f"Individuals 0-1 specialize in cases 0-3")
    print(f"Individuals 2-3 specialize in cases 4-7") 
    print(f"Individuals 4-5 specialize in cases 8-11")
    print()
    
    # Show the fitness matrix structure
    print("Fitness matrix (rounded):")
    print(np.round(fitness_matrix, 1))
    print()
    
    # Compare case selection diversity
    downsample_size = 4
    informed_counts, random_counts = analyze_case_selection_diversity(
        fitness_matrix, downsample_size=downsample_size
    )
    
    print(f"Case selection frequency over 100 trials (downsample_size={downsample_size}):")
    print("Case:    ", "".join(f"{i:4d}" for i in range(12)))
    print("Informed:", "".join(f"{int(count):4d}" for count in informed_counts))
    print("Random:  ", "".join(f"{int(count):4d}" for count in random_counts))
    print()
    
    # Calculate diversity metrics
    informed_entropy = -np.sum(informed_counts * np.log(informed_counts + 1e-10))
    random_entropy = -np.sum(random_counts * np.log(random_counts + 1e-10))
    
    print(f"Selection entropy (higher = more diverse):")
    print(f"  Informed: {informed_entropy:.2f}")
    print(f"  Random:   {random_entropy:.2f}")
    print()
    
    # Demonstrate actual selection with both methods
    print("Example selections with different methods:")
    print("-" * 40)
    
    seed = 42
    num_selected = 3
    
    # Full lexicase (no downsampling)
    full_selected = lexicase_selection(fitness_matrix, num_selected, seed=seed)
    print(f"Full lexicase:      {full_selected}")
    
    # Random downsampling
    random_selected = downsample_lexicase_selection(
        fitness_matrix, num_selected, downsample_size, seed=seed
    )
    print(f"Random downsample:  {random_selected}")
    
    # Informed downsampling
    informed_selected = informed_downsample_lexicase_selection(
        fitness_matrix, num_selected, downsample_size, seed=seed, sample_rate=0.5
    )
    print(f"Informed downsample: {informed_selected}")
    print()
    
    # Show selected individuals' specializations
    print("Selected individuals and their strengths:")
    for method, selected in [("Full", full_selected), 
                           ("Random", random_selected), 
                           ("Informed", informed_selected)]:
        print(f"{method:8s}: ", end="")
        for idx in selected:
            # Find which cases this individual is best at
            best_cases = np.where(fitness_matrix[idx] > 7)[0]
            if len(best_cases) > 0:
                print(f"Ind{idx}(cases {best_cases[0]}-{best_cases[-1]}) ", end="")
            else:
                print(f"Ind{idx}(generalist) ", end="")
        print()
    
    print()
    print("Key observations:")
    print("- Informed downsampling tends to select more diverse test cases")
    print("- This can lead to better representation of different specialist types") 
    print("- The method adapts to the population's current capabilities")

if __name__ == "__main__":
    demonstrate_informed_downsample()