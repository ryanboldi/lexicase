"""A whole GA, selection included, jitted and vmapped over independent runs in JAX."""

import time

import jax
import jax.numpy as jnp

from lexicase.jax_impl import jax_lexicase_selection

N_CASES = 64
POP_SIZE = 64
GENERATIONS = 60
N_RUNS = 64


def step(carry, _):
    population, key = carry
    key, select_key, pair_key, cross_key = jax.random.split(key, 4)

    parents = jax_lexicase_selection(population.astype(jnp.float32), POP_SIZE, select_key)
    mothers = population[parents]
    fathers = population[jax.random.permutation(pair_key, parents)]
    crossover = jax.random.uniform(cross_key, population.shape) < 0.5
    population = jnp.where(crossover, mothers, fathers)

    return (population, key), population.sum(axis=1).max()


@jax.jit
def evolve(key):
    """One full run. Returns the best case count reached in each generation."""
    init_key, run_key = jax.random.split(key)
    population = (jax.random.uniform(init_key, (POP_SIZE, N_CASES)) < 0.5).astype(jnp.int8)
    _, best_per_generation = jax.lax.scan(
        step, (population, run_key), None, length=GENERATIONS
    )
    return best_per_generation


def main():
    print(f"{N_RUNS} independent runs, population {POP_SIZE}, {N_CASES} cases, "
          f"{GENERATIONS} generations")
    print("The generation loop is one lax.scan inside one jit, and the runs are vmapped.")
    print()

    keys = jax.random.split(jax.random.PRNGKey(0), N_RUNS)
    evolve_many = jax.jit(jax.vmap(evolve))

    start = time.perf_counter()
    curves = evolve_many(keys).block_until_ready()
    compile_and_run = time.perf_counter() - start

    start = time.perf_counter()
    curves = evolve_many(keys).block_until_ready()
    run_only = time.perf_counter() - start

    solved = (curves == N_CASES).any(axis=1)
    first_solve = jnp.where(solved, jnp.argmax(curves == N_CASES, axis=1), GENERATIONS)

    print(f"compile plus first run  {compile_and_run:8.3f} s")
    print(f"second run              {run_only:8.3f} s  "
          f"({1000 * run_only / N_RUNS:.1f} ms per run)")
    print()
    print(f"solved {int(solved.sum())}/{N_RUNS} runs")
    print(f"median generation at first solve {float(jnp.median(first_solve[solved])):.0f}")
    print()
    print("Mean best case count by generation:")
    means = curves.mean(axis=0)
    for generation in range(0, GENERATIONS, 10):
        print(f"  gen {generation:3d}  {float(means[generation]):.1f} / {N_CASES}")


if __name__ == "__main__":
    main()
