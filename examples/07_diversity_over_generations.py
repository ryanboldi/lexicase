"""Diversity over generations under lexicase, downsampled lexicase, and tournament, with a plot."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines import tournament_selection  # noqa: E402
from lexicase import downsample_lexicase_selection, lexicase_selection  # noqa: E402

N_CASES = 150
POP_SIZE = 60
GENERATIONS = 60
N_RUNS = 8
OUTPUT = Path(__file__).parent / "diversity_over_generations.png"


def diversity(population):
    """Mean pairwise Hamming distance, normalized. Zero once every genome agrees."""
    frequency = population.mean(axis=0)
    return float(np.mean(2 * frequency * (1 - frequency)))


def run(select, seed):
    rng = np.random.default_rng(seed)
    population = (rng.random((POP_SIZE, N_CASES)) < 0.5).astype(np.int8)
    diversities, best = [], []

    for generation in range(GENERATIONS):
        diversities.append(diversity(population))
        best.append(population.sum(axis=1).max() / N_CASES)
        parents = select(population.astype(float), POP_SIZE, seed * 1000 + generation)
        mothers = population[parents]
        fathers = population[rng.permutation(parents)]
        population = np.where(rng.random((POP_SIZE, N_CASES)) < 0.5, mothers, fathers)

    return np.array(diversities), np.array(best)


def main():
    methods = {
        "lexicase": lambda f, n, seed: lexicase_selection(f, n, seed=seed),
        "downsample 10": lambda f, n, seed: downsample_lexicase_selection(
            f, n, 10, seed=seed
        ),
        "tournament (size 3)": lambda f, n, seed: tournament_selection(
            f, n, tournament_size=3, seed=seed
        ),
    }

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
    generations = np.arange(GENERATIONS)

    for name, select in methods.items():
        runs = [run(select, seed) for seed in range(N_RUNS)]
        diversities = np.stack([curve for curve, _ in runs])
        best = np.stack([curve for _, curve in runs])

        mean, spread = diversities.mean(axis=0), diversities.std(axis=0)
        top.plot(generations, mean, label=name)
        top.fill_between(generations, mean - spread, mean + spread, alpha=0.15)
        bottom.plot(generations, best.mean(axis=0), label=name)

        print(f"{name:22s} diversity gen 0 {mean[0]:.3f}, gen 10 {mean[10]:.3f}, "
              f"gen 20 {mean[20]:.3f}, gen {GENERATIONS - 1} {mean[-1]:.3f}   "
              f"best cases passed at the end {best.mean(axis=0)[-1]:.2f}")

    top.set_ylabel("mean pairwise Hamming distance")
    top.set_title(f"{N_CASES} cases, population {POP_SIZE}, {N_RUNS} runs")
    top.legend()
    bottom.set_xlabel("generation")
    bottom.set_ylabel("best fraction of cases passed")
    figure.tight_layout()
    figure.savefig(OUTPUT, dpi=140)
    print()
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
