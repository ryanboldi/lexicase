"""Population diversity over generations under lexicase, downsampled lexicase, and tournament."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines import tournament_selection  # noqa: E402

from lexicase import downsample_lexicase_selection, lexicase_selection  # noqa: E402

N_CASES = 60
POP_SIZE = 60
GENERATIONS = 60
N_RUNS = 10
OUTPUT = Path(__file__).parent / "diversity_over_generations.png"


def behavioural_diversity(population):
    """Fraction of the population that has a distinct pass/fail signature."""
    return len(np.unique(population, axis=0)) / len(population)


def run(select, seed):
    rng = np.random.default_rng(seed)
    population = (rng.random((POP_SIZE, N_CASES)) < 0.5).astype(np.int8)
    curve = []

    for generation in range(GENERATIONS):
        curve.append(behavioural_diversity(population))
        parents = select(population.astype(float), POP_SIZE, seed * 1000 + generation)
        mothers = population[parents]
        fathers = population[rng.permutation(parents)]
        population = np.where(rng.random((POP_SIZE, N_CASES)) < 0.5, mothers, fathers)

    return np.array(curve)


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

    figure, axes = plt.subplots(figsize=(7, 4.5))
    for name, select in methods.items():
        curves = np.stack([run(select, seed) for seed in range(N_RUNS)])
        mean = curves.mean(axis=0)
        spread = curves.std(axis=0)
        generations = np.arange(GENERATIONS)
        axes.plot(generations, mean, label=name)
        axes.fill_between(generations, mean - spread, mean + spread, alpha=0.15)
        print(f"{name:22s} diversity at gen 0 {mean[0]:.2f}, at gen "
              f"{GENERATIONS - 1} {mean[-1]:.2f}")

    axes.set_xlabel("generation")
    axes.set_ylabel("fraction of distinct behaviours")
    axes.set_title(f"Behavioural diversity, mean and standard deviation over {N_RUNS} runs")
    axes.set_ylim(0, 1.05)
    axes.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT, dpi=140)
    print()
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
