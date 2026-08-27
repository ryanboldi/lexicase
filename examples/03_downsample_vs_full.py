"""Downsampled versus full lexicase: solve rate and selection wall clock side by side."""

import time

import numpy as np

from lexicase import downsample_lexicase_selection, lexicase_selection

N_CASES = 100
POP_SIZE = 60
GENERATIONS = 100
N_RUNS = 15


def run(select, seed):
    """Crossover-only GA. Returns (solved, generations used, seconds spent selecting)."""
    rng = np.random.default_rng(seed)
    population = (rng.random((POP_SIZE, N_CASES)) < 0.5).astype(np.int8)
    seconds = 0.0

    for generation in range(GENERATIONS):
        if population.sum(axis=1).max() == N_CASES:
            return True, generation, seconds

        start = time.perf_counter()
        parents = select(population.astype(float), POP_SIZE, seed * 1000 + generation)
        seconds += time.perf_counter() - start

        mothers = population[parents]
        fathers = population[rng.permutation(parents)]
        population = np.where(rng.random((POP_SIZE, N_CASES)) < 0.5, mothers, fathers)

    return False, GENERATIONS, seconds


def main():
    methods = {
        "full lexicase": lambda f, n, seed: lexicase_selection(f, n, seed=seed),
        "downsample 25": lambda f, n, seed: downsample_lexicase_selection(
            f, n, 25, seed=seed
        ),
        "downsample 10": lambda f, n, seed: downsample_lexicase_selection(
            f, n, 10, seed=seed
        ),
        "downsample 5": lambda f, n, seed: downsample_lexicase_selection(
            f, n, 5, seed=seed
        ),
    }

    print(f"{N_CASES} cases, population {POP_SIZE}, up to {GENERATIONS} generations, "
          f"{N_RUNS} runs")
    print()
    header = f"{'method':16s} {'solved':>8s} {'median gens':>12s} {'selection ms/run':>18s}"
    print(header)
    print("-" * len(header))

    for name, select in methods.items():
        results = [run(select, seed) for seed in range(N_RUNS)]
        solved = [generation for ok, generation, _ in results if ok]
        milliseconds = 1000 * np.mean([seconds for _, _, seconds in results])
        median = f"{np.median(solved):.0f}" if solved else "n/a"
        print(
            f"{name:16s} {len(solved):>4d}/{N_RUNS:<3d} {median:>12s} {milliseconds:>18.1f}"
        )

    print()
    print("Selection time falls roughly with the number of cases used per event.")
    print("Very small downsamples give that back by needing more generations.")


if __name__ == "__main__":
    main()
