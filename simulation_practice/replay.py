"""
Replay the best evolved gait in MuJoCo's interactive viewer.

Requirements
------------
  A display (X11 / Wayland) is needed.  On a headless server, prefix with:
    MUJOCO_GL=osmesa python replay.py   (software rendering)

Usage
-----
  # Replay best genome from results.pkl (created by evolve.py)
  python replay.py

  # Replay from a checkpoint instead
  python replay.py --pkl checkpoint.pkl

  # Show the top-N hall-of-fame entries one after another
  python replay.py --top 3

  # Play a specific genome given as raw gene values on the command line
  python replay.py --genome 0.82 0.41 1.57  0.70 0.60 0.00  ...  (24 values)
"""

import math
import os
import pickle
import time
import argparse

import mujoco
import mujoco.viewer

# Import shared constants from evolve.py (same directory)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evolve import decode, XML_PATH, DT, NUM_ACTUATORS


# ── Core replay function ──────────────────────────────────────────────────────

def replay(genome, duration: float = 12.0, real_time: bool = True):
    """
    Simulate `genome` for `duration` seconds and render in an interactive window.

    Parameters
    ----------
    genome    : list of 24 raw gene values in [0, 1]
    duration  : how many simulated seconds to run
    real_time : if True, sleep to match wall-clock time
    """
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    params = decode(list(genome))
    steps  = int(duration / DT)

    print(f"\nReplaying genome for {duration} s …  (close the viewer window to quit)")
    print(f"Decoded controller params:")
    for i, (amp, freq, phase) in enumerate(params):
        print(f"  actuator {i}: amp={amp:.3f}  freq={freq:.2f} Hz  phase={math.degrees(phase):.1f}°")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        for step in range(steps):
            if not viewer.is_running():
                break

            t = step * DT
            for i, (amp, freq, phase) in enumerate(params):
                data.ctrl[i] = amp * math.sin(2.0 * math.pi * freq * t + phase)

            mujoco.mj_step(model, data)
            viewer.sync()

            # Throttle to real time so the motion is easy to watch
            if real_time:
                wall_elapsed  = time.time() - t0
                sim_elapsed   = (step + 1) * DT
                sleep_needed  = sim_elapsed - wall_elapsed
                if sleep_needed > 0:
                    time.sleep(sleep_needed)

    print(f"\nFinal torso position:  x = {data.qpos[0]:.3f} m")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Replay evolved gaits in MuJoCo viewer")
    p.add_argument("--pkl",      default="results.pkl",
                   help="Pickle file produced by evolve.py (default: results.pkl)")
    p.add_argument("--top",      type=int, default=1,
                   help="Replay the top-N hall-of-fame genomes in sequence (default: 1)")
    p.add_argument("--duration", type=float, default=12.0,
                   help="Replay duration in seconds (default: 12)")
    p.add_argument("--no-realtime", action="store_true",
                   help="Run as fast as possible instead of real-time")
    p.add_argument("--genome",   nargs="+", type=float,
                   help=f"Provide {NUM_ACTUATORS * 3} raw gene values directly")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.genome:
        # Manual genome from command line
        genome = args.genome
        if len(genome) != NUM_ACTUATORS * 3:
            raise ValueError(
                f"Expected {NUM_ACTUATORS * 3} gene values, got {len(genome)}"
            )
        replay(genome, duration=args.duration, real_time=not args.no_realtime)

    else:
        # Load from pickle
        pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.pkl)
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"'{pkl_path}' not found.  Run evolve.py first, or pass --genome."
            )

        with open(pkl_path, "rb") as f:
            results = pickle.load(f)

        hof = results["hof"]
        n   = min(args.top, len(hof))

        for rank, ind in enumerate(hof[:n]):
            fitness = ind.fitness.values[0]
            print(f"\n{'=' * 50}")
            print(f"Hall-of-Fame rank {rank + 1}  |  fitness = {fitness:.3f} m")
            print(f"{'=' * 50}")
            replay(list(ind), duration=args.duration, real_time=not args.no_realtime)
