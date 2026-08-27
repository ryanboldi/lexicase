"""Lexicase as a selection operator over a reward tensor that stays on the GPU."""

import time

import torch

from lexicase import lexicase_selection

N_SAMPLES = 512
N_OBJECTIVES = 64
GENERATIONS = 20
SEED = 0


def fake_rollout_rewards(policy_params, generator, device):
    """Stand-in for a rollout: per-sample, per-objective rewards from the trainer."""
    noise = torch.randn(
        (N_SAMPLES, N_OBJECTIVES), generator=generator, device=device
    )
    return policy_params[:, None] + noise


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(SEED)
    print(f"device {device}, {N_SAMPLES} samples, {N_OBJECTIVES} objectives")
    print()

    policy_params = torch.randn(N_SAMPLES, generator=generator, device=device)

    for generation in range(GENERATIONS):
        rewards = fake_rollout_rewards(policy_params, generator, device)

        # The reward tensor never leaves the device, and neither does the result.
        parents = lexicase_selection(rewards, N_SAMPLES, seed=SEED + generation)
        assert parents.device == rewards.device

        mutation = 0.1 * torch.randn(
            N_SAMPLES, generator=generator, device=device
        )
        policy_params = policy_params.index_select(0, parents) + mutation

        if generation % 5 == 0:
            print(
                f"gen {generation:2d}  mean reward "
                f"{float(rewards.mean()):+.4f}  unique parents "
                f"{int(torch.unique(parents).numel())}"
            )

    print()
    rewards = fake_rollout_rewards(policy_params, generator, device)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        lexicase_selection(rewards, N_SAMPLES, seed=SEED)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"selection cost: {1000 * (time.perf_counter() - start) / 10:.2f} ms per call")
    print()
    print("Nothing in the selection call synchronizes with the host, so it can sit")
    print("inside a training step without stalling the pipeline. Run")
    print("benchmarks/torch_cuda_check.py to verify that on your own device.")


if __name__ == "__main__":
    main()
