"""Symbolic regression with epsilon lexicase: fit a polynomial to noisy data."""

import numpy as np

from lexicase import epsilon_lexicase_selection

SEED = 0
POP_SIZE = 200
GENERATIONS = 60
DEGREE = 3
N_CASES = 40

TRUE_COEFFICIENTS = np.array([1.5, -2.0, 0.0, 0.75])


def evaluate(population, x, y):
    """Negative absolute error per case, so higher is better."""
    powers = np.vander(x, DEGREE + 1, increasing=True)
    predictions = population @ powers.T
    return -np.abs(predictions - y[None, :])


def main():
    rng = np.random.default_rng(SEED)

    x = np.linspace(-2.0, 2.0, N_CASES)
    y = np.vander(x, DEGREE + 1, increasing=True) @ TRUE_COEFFICIENTS
    y = y + rng.normal(0.0, 0.05, size=N_CASES)

    population = rng.normal(0.0, 2.0, size=(POP_SIZE, DEGREE + 1))

    for generation in range(GENERATIONS):
        fitness = evaluate(population, x, y)
        parents = epsilon_lexicase_selection(
            fitness, num_selected=POP_SIZE, seed=SEED + generation, elitism=1
        )
        step = 0.3 * (0.95**generation)
        children = population[parents] + rng.normal(0.0, step, size=population.shape)
        children[0] = population[parents[0]]
        population = children

        if generation % 10 == 0 or generation == GENERATIONS - 1:
            errors = np.abs(evaluate(population, x, y)).mean(axis=1)
            best = int(np.argmin(errors))
            print(
                f"gen {generation:3d}  best mean abs error {errors[best]:.4f}  "
                f"coefficients {np.round(population[best], 3)}"
            )

    errors = np.abs(evaluate(population, x, y)).mean(axis=1)
    best = population[int(np.argmin(errors))]
    print()
    print(f"true coefficients  {TRUE_COEFFICIENTS}")
    print(f"found coefficients {np.round(best, 3)}")


if __name__ == "__main__":
    main()
