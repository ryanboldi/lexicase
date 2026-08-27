"""Informed downsampling picks test cases that disagree with each other, not just random ones."""

import numpy as np

from lexicase import (
    downsample_lexicase_selection,
    informed_downsample_lexicase_selection,
    lexicase_selection,
)
from lexicase.numpy_impl import _compute_case_distances, _farthest_first_traversal

SEED = 42
DOWNSAMPLE_SIZE = 3
SAMPLE_RATE = 0.5
N_TRIALS = 500
GROUP_SIZE = 4


def specialist_population():
    """Six individuals in three specialist pairs, twelve cases in three redundant groups."""
    rng = np.random.default_rng(SEED)
    fitness = np.empty((6, 12))
    for group in range(3):
        strong = slice(2 * group, 2 * group + 2)
        fitness[strong, :] = rng.uniform(1, 3, (2, 12))
        fitness[strong, 4 * group : 4 * group + 4] = rng.uniform(8, 10, (2, 4))
    return fitness


def groups_covered(fitness):
    """How many of the three case groups each method's downsample touches."""
    n_sampled = max(1, int(fitness.shape[0] * SAMPLE_RATE))
    informed, uniform = [], []

    for trial in range(N_TRIALS):
        rng = np.random.default_rng(trial)
        sample = rng.choice(fitness.shape[0], size=n_sampled, replace=False)
        distances = _compute_case_distances(fitness, sample)
        picks = _farthest_first_traversal(distances, DOWNSAMPLE_SIZE, rng)
        informed.append(len({case // GROUP_SIZE for case in picks}))

        rng = np.random.default_rng(trial)
        picks = rng.choice(fitness.shape[1], size=DOWNSAMPLE_SIZE, replace=False)
        uniform.append(len({case // GROUP_SIZE for case in picks}))

    return np.array(informed), np.array(uniform)


def main():
    fitness = specialist_population()
    print("Individuals 0-1 are strong on cases 0-3, 2-3 on cases 4-7, 4-5 on cases 8-11.")
    print("Cases inside a group carry nearly the same information, so a downsample that")
    print("spends two of its three slots on one group has wasted a slot.")
    print()

    informed, uniform = groups_covered(fitness)
    print(f"Distinct case groups covered, {N_TRIALS} trials, downsample size "
          f"{DOWNSAMPLE_SIZE} out of 12 cases, best possible 3:")
    print(f"  informed  {informed.mean():.2f}   "
          f"all three groups in {100 * (informed == 3).mean():.0f}% of trials")
    print(f"  uniform   {uniform.mean():.2f}   "
          f"all three groups in {100 * (uniform == 3).mean():.0f}% of trials")
    print()

    print("Selections with each method (seed 42, 6 parents):")
    print(f"  full lexicase       {lexicase_selection(fitness, 6, seed=SEED)}")
    print(
        "  uniform downsample  "
        f"{downsample_lexicase_selection(fitness, 6, DOWNSAMPLE_SIZE, seed=SEED)}"
    )
    informed = informed_downsample_lexicase_selection(
        fitness, 6, DOWNSAMPLE_SIZE, seed=SEED, sample_rate=SAMPLE_RATE
    )
    print(f"  informed downsample {informed}")


if __name__ == "__main__":
    main()
