"""An uncompromising problem where lexicase selection solves far more runs than tournament."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines import tournament_selection  # noqa: E402

from lexicase import lexicase_selection  # noqa: E402

N_CASES = 80
POP_SIZE = 40
GENERATIONS = 80
N_RUNS = 30


def run(select, seed):
    """Crossover-only GA on a problem where every case has to be satisfied at once."""
    rng = np.random.default_rng(seed)
    population = (rng.random((POP_SIZE, N_CASES)) < 0.5).astype(np.int8)

    for generation in range(GENERATIONS):
        if population.sum(axis=1).max() == N_CASES:
            return True, generation

        parents = select(population.astype(float), POP_SIZE, seed * 1000 + generation)
        mothers = population[parents]
        fathers = population[rng.permutation(parents)]
        crossover_mask = rng.random((POP_SIZE, N_CASES)) < 0.5
        population = np.where(crossover_mask, mothers, fathers)

    return False, GENERATIONS


def main():
    methods = {
        "lexicase": lambda f, n, seed: lexicase_selection(f, n, seed=seed),
        "tournament (size 3)": lambda f, n, seed: tournament_selection(
            f, n, tournament_size=3, seed=seed
        ),
    }

    print(f"{N_CASES} cases, population {POP_SIZE}, {GENERATIONS} generations, "
          f"{N_RUNS} runs, crossover only")
    print("Every case must be passed at once, so no single lineage can get there alone.")
    print()
    print(f"{'method':22s} {'solved':>8s} {'median gens to solve':>22s}")

    for name, select in methods.items():
        results = [run(select, seed) for seed in range(N_RUNS)]
        solved = [generation for ok, generation in results if ok]
        median = f"{np.median(solved):.0f}" if solved else "n/a"
        print(f"{name:22s} {len(solved):>4d}/{N_RUNS:<3d} {median:>22s}")

    print()
    print("Tournament selects on the total number of passed cases, so it converges on")
    print("one lineage and loses the rare alleles the remaining cases need. Lexicase")
    print("keeps specialists alive, and crossover assembles them.")


if __name__ == "__main__":
    main()
