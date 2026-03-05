"""
Evolutionary locomotion for a MuJoCo quadruped using DEAP.

Controller
----------
Open-loop sinusoidal: for each of the 8 actuators,

    u_i(t) = A_i * sin(2π * f_i * t + φ_i)

Gene encoding (24 floats, all normalised to [0, 1]):
  For actuator i, genes at positions [3i, 3i+1, 3i+2] encode:
    amplitude  → scaled to [0.3, 1.0]
    frequency  → scaled to [0.5, 3.0] Hz
    phase      → scaled to [0, 2π] rad

Fitness
-------
  x-distance (metres) the torso travels in SIM_DURATION seconds.
  An early-termination penalty applies if the robot falls over.

Usage
-----
  python evolve.py                          # defaults
  python evolve.py --pop 80 --gens 100      # larger run
  python evolve.py --workers 4              # parallel evaluation
"""

import math
import os
import pickle
import random
import argparse
import multiprocessing

import numpy as np
import mujoco
from deap import algorithms, base, creator, tools

# ── Paths ────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(_DIR, "crawler.xml")

# ── Simulation parameters ────────────────────────────────────────────────────
SIM_DURATION   = 5.0    # seconds per fitness evaluation
DT             = 0.005  # must match <option timestep> in crawler.xml
NUM_ACTUATORS  = 8
MIN_TORSO_Z    = 0.20   # below this the robot has "fallen"; end early

# ── Gene → physical parameter ranges ────────────────────────────────────────
AMP_RANGE   = (0.3, 1.0)
FREQ_RANGE  = (0.5, 3.0)
PHASE_RANGE = (0.0, 2.0 * math.pi)

GENES_PER_ACT = 3                        # amplitude, frequency, phase
GENOME_LEN    = NUM_ACTUATORS * GENES_PER_ACT   # 24


# ── Gene helpers ─────────────────────────────────────────────────────────────

def _scale(raw: float, lo: float, hi: float) -> float:
    """Map a raw gene in [0, 1] to the physical range [lo, hi]."""
    return lo + raw * (hi - lo)


def decode(genome) -> list:
    """
    Return a list of (amplitude, frequency_hz, phase_rad) tuples,
    one per actuator, decoded from the raw [0, 1] genome.
    """
    params = []
    for i in range(NUM_ACTUATORS):
        b = i * GENES_PER_ACT
        amp   = _scale(genome[b],     *AMP_RANGE)
        freq  = _scale(genome[b + 1], *FREQ_RANGE)
        phase = _scale(genome[b + 2], *PHASE_RANGE)
        params.append((amp, freq, phase))
    return params


# ── Fitness evaluation ────────────────────────────────────────────────────────

def evaluate(individual) -> tuple:
    """
    Simulate the robot with the given genome and return (x_distance,).

    This function is designed to be safe for use in a multiprocessing pool
    (each worker creates its own MjModel / MjData; no shared state).
    """
    model  = mujoco.MjModel.from_xml_path(XML_PATH)
    data   = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    params = decode(list(individual))
    steps  = int(SIM_DURATION / DT)

    for step in range(steps):
        t = step * DT
        for i, (amp, freq, phase) in enumerate(params):
            data.ctrl[i] = amp * math.sin(2.0 * math.pi * freq * t + phase)
        mujoco.mj_step(model, data)

        # Early exit if the robot collapses
        if data.qpos[2] < MIN_TORSO_Z:
            break

    x_distance = float(data.qpos[0])
    return (x_distance,)


# ── DEAP setup ────────────────────────────────────────────────────────────────

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float",  random.random)
toolbox.register("individual",  tools.initRepeat,
                 creator.Individual, toolbox.attr_float, n=GENOME_LEN)
toolbox.register("population",  tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate",    evaluate)
toolbox.register("mate",        tools.cxBlend, alpha=0.5)
toolbox.register("mutate",      tools.mutGaussian, mu=0, sigma=0.08, indpb=0.2)
toolbox.register("select",      tools.selTournament, tournsize=3)


def _clamp(ind) -> None:
    """In-place: keep all genes in [0, 1] after crossover / mutation."""
    for i in range(len(ind)):
        ind[i] = max(0.0, min(1.0, ind[i]))


# ── Main evolution loop ───────────────────────────────────────────────────────

def run_evolution(
    pop_size: int = 50,
    n_generations: int = 50,
    cx_prob: float = 0.7,
    mut_prob: float = 0.2,
    workers: int = 1,
    checkpoint_every: int = 10,
    checkpoint_path: str = "checkpoint.pkl",
    results_path: str = "results.pkl",
):
    """
    Run the genetic algorithm and save the best results to disk.

    Parameters
    ----------
    pop_size        : number of individuals per generation
    n_generations   : number of generations to evolve
    cx_prob         : probability of crossover between two offspring
    mut_prob        : probability of mutating an individual
    workers         : parallel workers for fitness evaluation (1 = serial)
    checkpoint_every: save a checkpoint every N generations
    checkpoint_path : path for intermediate checkpoints
    results_path    : path for the final results file
    """

    # Optionally use a multiprocessing pool for parallel evaluation
    pool = None
    if workers > 1:
        pool = multiprocessing.Pool(processes=workers)
        toolbox.register("map", pool.map)
        print(f"Using {workers} parallel workers.")

    pop  = toolbox.population(n=pop_size)
    hof  = tools.HallOfFame(5)           # keep 5 best individuals ever seen

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max",  np.max)
    stats.register("mean", np.mean)
    stats.register("std",  np.std)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals", "max", "mean", "std"]

    # ── Generation 0: evaluate initial population ──────────────────────────
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    print(logbook.stream)

    # ── Main loop ──────────────────────────────────────────────────────────
    for gen in range(1, n_generations + 1):

        # Select and clone offspring
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(c1, c2)
                _clamp(c1)
                _clamp(c2)
                del c1.fitness.values
                del c2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                _clamp(mutant)
                del mutant.fitness.values

        # Evaluate only individuals whose fitness is stale
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid), **record)
        print(logbook.stream)

        # Periodic checkpoint
        if gen % checkpoint_every == 0:
            _save(checkpoint_path, pop, hof, logbook, gen)
            print(f"  [checkpoint → {checkpoint_path}]")

    # ── Finalise ───────────────────────────────────────────────────────────
    if pool is not None:
        pool.close()
        pool.join()

    _save(results_path, pop, hof, logbook, n_generations)
    print(f"\nEvolution complete.  Results saved to '{results_path}'.")
    print(f"Best fitness : {hof[0].fitness.values[0]:.3f} m  (forward distance)")
    return pop, hof, logbook


def _save(path, pop, hof, logbook, gen):
    with open(path, "wb") as f:
        pickle.dump({"population": pop, "hof": hof, "logbook": logbook, "gen": gen}, f)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Evolve a quadruped gait with DEAP + MuJoCo"
    )
    p.add_argument("--pop",     type=int,   default=50,  help="Population size (default: 50)")
    p.add_argument("--gens",    type=int,   default=50,  help="Generations (default: 50)")
    p.add_argument("--cx",      type=float, default=0.7, help="Crossover prob (default: 0.7)")
    p.add_argument("--mut",     type=float, default=0.2, help="Mutation prob  (default: 0.2)")
    p.add_argument("--workers", type=int,   default=1,
                   help="Parallel eval workers (default: 1; use # of CPU cores for speed)")
    p.add_argument("--checkpoint-every", type=int, default=10,
                   help="Save checkpoint every N gens (default: 10)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evolution(
        pop_size        = args.pop,
        n_generations   = args.gens,
        cx_prob         = args.cx,
        mut_prob        = args.mut,
        workers         = args.workers,
        checkpoint_every= args.checkpoint_every,
    )
