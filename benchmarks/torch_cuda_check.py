"""Run every Torch kernel on CUDA with sync debug mode on, to prove none of them stall."""

import sys

import torch

import lexicase as lx

N_INDIVIDUALS = 512
N_CASES = 128
SEED = 0


def main():
    if not torch.cuda.is_available():
        print("no CUDA device, nothing to check")
        return 0

    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
    print()

    generator = torch.Generator(device="cuda").manual_seed(SEED)
    fitness = torch.rand(
        (N_INDIVIDUALS, N_CASES), generator=generator, device="cuda"
    )
    weights = torch.rand(N_CASES, generator=generator, device="cuda") + 0.1
    epsilon = torch.rand(N_CASES, generator=generator, device="cuda") * 0.1
    n = N_INDIVIDUALS

    calls = {
        "lexicase": lambda: lx.lexicase_selection(fitness, n, seed=SEED),
        "lexicase elitism=8": lambda: lx.lexicase_selection(
            fitness, n, seed=SEED, elitism=8
        ),
        "lexicase weighted": lambda: lx.lexicase_selection(
            fitness, n, seed=SEED, case_weights=weights
        ),
        "epsilon scalar": lambda: lx.epsilon_lexicase_selection(
            fitness, n, 0.05, seed=SEED
        ),
        "epsilon per-case": lambda: lx.epsilon_lexicase_selection(
            fitness, n, epsilon, seed=SEED
        ),
        "epsilon static": lambda: lx.epsilon_lexicase_selection(
            fitness, n, 0.05, seed=SEED, mode="static"
        ),
        "epsilon dynamic": lambda: lx.epsilon_lexicase_selection(
            fitness, n, seed=SEED, mode="dynamic"
        ),
        "epsilon MAD default": lambda: lx.epsilon_lexicase_selection(
            fitness, n, seed=SEED
        ),
        "downsample": lambda: lx.downsample_lexicase_selection(
            fitness, n, 16, seed=SEED
        ),
        "informed downsample": lambda: lx.informed_downsample_lexicase_selection(
            fitness, n, 16, seed=SEED, sample_rate=0.1, threshold=0.5
        ),
        "batch": lambda: lx.batch_lexicase_selection(fitness, n, 8, seed=SEED),
        "cohort": lambda: lx.cohort_lexicase_selection(fitness, n, 4, seed=SEED),
        "dalex": lambda: lx.dalex_selection(fitness, n, seed=SEED),
    }

    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    rows = []
    for name, call in calls.items():
        try:
            result = call()
            rows.append((name, str(result.device), "no sync"))
        except Exception as exc:
            rows.append((name, "-", f"{type(exc).__name__}: {exc}"))
    torch.cuda.set_sync_debug_mode("default")

    failures = 0
    for name, device, note in rows:
        print(f"{name:22s} {device:8s} {note}")
        failures += note != "no sync"

    print()
    print("plexicase is documented to synchronize, since finding the Pareto set")
    print("boundaries needs a data-dependent candidate count:")
    print(f"  plexicase result device {lx.plexicase_selection(fitness, n, seed=SEED).device}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
