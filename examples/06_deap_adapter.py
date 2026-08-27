"""Feed a DEAP-style population into lexicase selection with a drop-in selector."""

import numpy as np

from lexicase import epsilon_lexicase_selection

SEED = 3


class Fitness:
    """Stands in for deap.base.Fitness. values holds one error per test case."""

    def __init__(self, values):
        self.values = tuple(values)


class Individual(list):
    """Stands in for a DEAP individual, which is a list with a fitness attached."""

    def __init__(self, genes, errors):
        super().__init__(genes)
        self.fitness = Fitness(errors)


def sel_lexicase(individuals, k, minimize=True, seed=None, **kwargs):
    """A DEAP-shaped selector: takes (individuals, k), returns k individuals.

    DEAP stores per-case values in `ind.fitness.values`. This library wants a
    (n_individuals, n_cases) matrix where higher is better, so error-valued
    fitness gets negated on the way in.
    """
    matrix = np.array([individual.fitness.values for individual in individuals], dtype=float)
    if minimize:
        matrix = -matrix
    chosen = epsilon_lexicase_selection(matrix, num_selected=k, seed=seed, **kwargs)
    return [individuals[index] for index in chosen]


def main():
    rng = np.random.default_rng(SEED)

    targets = np.linspace(-1.0, 1.0, 12)
    population = []
    for _ in range(30):
        genes = rng.normal(0.0, 1.0, size=3)
        predictions = genes[0] + genes[1] * targets + genes[2] * targets**2
        errors = np.abs(predictions - targets**2)
        population.append(Individual(genes, errors))

    print("A DEAP-style population: each individual carries one error per case.")
    print(f"population size {len(population)}, cases per individual "
          f"{len(population[0].fitness.values)}")
    print()

    parents = sel_lexicase(population, k=10, seed=SEED)
    print("Selected 10 parents with epsilon lexicase.")
    for individual in parents[:5]:
        mean_error = np.mean(individual.fitness.values)
        print(f"  genes {np.round(np.asarray(individual), 3)}  mean error {mean_error:.3f}")
    print("  ...")
    print()

    population_error = np.mean([np.mean(i.fitness.values) for i in population])
    parent_error = np.mean([np.mean(i.fitness.values) for i in parents])
    print(f"mean error across the population {population_error:.3f}")
    print(f"mean error across the parents    {parent_error:.3f}")
    print()
    print("To use this with real DEAP: toolbox.register('select', sel_lexicase).")
    print("The same shape works for any framework that exposes per-case errors.")


if __name__ == "__main__":
    main()
