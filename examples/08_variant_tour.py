"""Every variant on one population, showing how selection pressure differs between them."""

import numpy as np

from lexicase import (
    batch_lexicase_selection,
    cohort_lexicase_selection,
    dalex_selection,
    downsample_lexicase_selection,
    epsilon_lexicase_selection,
    lexicase_selection,
    plexicase_probabilities,
    plexicase_selection,
)

SEED = 11
N_DRAWS = 20000


def population():
    """A two-case elite, two specialists, a dominated elite, a near-miss, a generalist, a dud."""
    return np.array(
        [
            [9.0, 9.0, 1.0, 1.0],
            [1.0, 1.0, 9.0, 1.0],
            [1.0, 1.0, 1.0, 9.0],
            [9.0, 1.0, 1.0, 1.0],
            [8.9, 8.9, 1.0, 1.0],
            [5.0, 5.0, 5.0, 5.0],
            [0.5, 0.5, 0.5, 0.5],
        ]
    )


def report(name, selected, n_individuals):
    counts = np.bincount(np.asarray(selected), minlength=n_individuals)
    shares = counts / counts.sum()
    print(f"{name:26s} " + " ".join(f"{share:5.3f}" for share in shares))


def main():
    fitness = population()
    n = len(fitness)

    print("Selection share per individual over", N_DRAWS, "draws.")
    print("0 is elite on cases 0 and 1. 1 and 2 are single-case specialists. 3 is elite")
    print("on case 0 but dominated by 0. 4 misses by 0.1. 5 is a generalist that is elite")
    print("nowhere. 6 is dominated by everyone.")
    print()
    print(f"{'method':26s} " + " ".join(f"ind{i:<2d}" for i in range(n)))
    print("-" * (26 + 6 * n))

    report("lexicase", lexicase_selection(fitness, N_DRAWS, seed=SEED), n)
    report(
        "epsilon (MAD, semi-dyn)",
        epsilon_lexicase_selection(fitness, N_DRAWS, seed=SEED),
        n,
    )
    report(
        "epsilon (dynamic)",
        epsilon_lexicase_selection(fitness, N_DRAWS, seed=SEED, mode="dynamic"),
        n,
    )
    report(
        "downsample 2",
        downsample_lexicase_selection(fitness, N_DRAWS, 2, seed=SEED),
        n,
    )
    report("batch 2", batch_lexicase_selection(fitness, N_DRAWS, 2, seed=SEED), n)
    report("batch 4 (elitist)", batch_lexicase_selection(fitness, N_DRAWS, 4, seed=SEED), n)
    report("cohort 2", cohort_lexicase_selection(fitness, N_DRAWS, 2, seed=SEED), n)
    report("plexicase (alpha=1)", plexicase_selection(fitness, N_DRAWS, seed=SEED), n)
    report(
        "plexicase (alpha=4)",
        plexicase_selection(fitness, N_DRAWS, seed=SEED, alpha=4.0),
        n,
    )
    report(
        "dalex (pressure 200)",
        dalex_selection(fitness, N_DRAWS, seed=SEED, particularity_pressure=200.0),
        n,
    )
    report(
        "dalex (pressure 1)",
        dalex_selection(fitness, N_DRAWS, seed=SEED, particularity_pressure=1.0),
        n,
    )
    report(
        "weighted case order",
        lexicase_selection(
            fitness, N_DRAWS, seed=SEED, case_weights=np.array([10.0, 1.0, 1.0, 1.0])
        ),
        n,
    )

    print()
    print("plexicase probabilities without drawing anything:")
    print("  ", np.round(plexicase_probabilities(fitness), 3))
    print()
    print("Reading the table:")
    print("  Individuals 5 and 6 are elite on no case, so lexicase can never pick them")
    print("  and plexicase gives them exactly zero.")
    print("  Epsilon lets individual 4, which misses by 0.1, back into contention.")
    print("  Cohorts are drawn once per call, so a single call gives lumpy shares: only")
    print("  the members of a cohort ever compete with each other.")
    print("  Raising the plexicase alpha concentrates mass on the most elite individual.")
    print("  High DALex pressure lands near plain lexicase, low pressure collapses onto")
    print("  the best mean fitness.")


if __name__ == "__main__":
    main()
