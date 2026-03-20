"""
Replay evolved or trained gaits in MuJoCo's interactive viewer.

Supports both result formats:
  - evolve.py   → results.pkl      (GA, contains hall-of-fame)
  - train_pg.py → pg_results.pkl   (policy gradient, contains policy weights)

The correct mode is detected automatically from the pickle contents.

Requirements
------------
  A display (X11 / Wayland) is needed.  On a headless server, prefix with:
    MUJOCO_GL=osmesa python replay.py   (software rendering)

Usage
-----
  # GA: replay best genome from results.pkl
  python replay.py

  # GA: top-3 hall-of-fame entries one after another
  python replay.py --top 3

  # PG: replay learned policy from pg_results.pkl
  python replay.py --pkl pg_results.pkl

  # Either: load a checkpoint instead of the final file
  python replay.py --pkl pg_checkpoint.pkl

  # GA only: supply raw gene values directly on the command line
  python replay.py --genome 0.82 0.41 1.57  0.70 0.60 0.00  ...  (24 values)
"""

import math
import os
import pickle
import time
import argparse

import numpy as np
import torch
import mujoco
import mujoco.viewer

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evolve   import decode, XML_PATH, DT, NUM_ACTUATORS
from train_pg import GaussianPolicy, get_obs, OBS_DIM, ACT_DIM


# ── GA replay ────────────────────────────────────────────────────────────────

def replay_genome(genome, duration: float = 12.0, real_time: bool = True):
    """Replay a sinusoidal GA genome in the viewer."""
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    params = decode(list(genome))
    steps  = int(duration / DT)

    print(f"\nReplaying GA genome for {duration} s  (close window to quit)")
    print("Decoded controller params:")
    for i, (amp, freq, phase) in enumerate(params):
        print(f"  actuator {i}: amp={amp:.3f}  freq={freq:.2f} Hz  "
              f"phase={math.degrees(phase):.1f}°")

    _run_viewer(model, data, steps, real_time,
                ctrl_fn=lambda step: [
                    amp * math.sin(2.0 * math.pi * freq * step * DT + phase)
                    for amp, freq, phase in params
                ])


# ── PG replay ─────────────────────────────────────────────────────────────────

def replay_policy(policy: GaussianPolicy, duration: float = 12.0,
                  real_time: bool = True):
    """Replay a trained Gaussian policy in the viewer (deterministic mean)."""
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    steps = int(duration / DT)

    print(f"\nReplaying PG policy for {duration} s  (close window to quit)")
    print("Running deterministically (using policy mean, no sampling).")

    def ctrl_fn(_step):
        obs  = torch.from_numpy(get_obs(data)).unsqueeze(0)
        with torch.no_grad():
            mean = policy.net(obs).squeeze(0)
        return mean.clamp(-1.0, 1.0).numpy()

    _run_viewer(model, data, steps, real_time, ctrl_fn)


# ── Shared viewer loop ────────────────────────────────────────────────────────

def _run_viewer(model, data, steps, real_time, ctrl_fn):
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        for step in range(steps):
            if not viewer.is_running():
                break

            ctrl = ctrl_fn(step)
            for i, c in enumerate(ctrl):
                data.ctrl[i] = c

            mujoco.mj_step(model, data)
            viewer.sync()

            if real_time:
                target = (step + 1) * DT
                elapsed = time.time() - t0
                if target > elapsed:
                    time.sleep(target - elapsed)

    print(f"Final torso position:  x = {data.qpos[0]:.3f} m")


# ── Auto-detect result type and dispatch ─────────────────────────────────────

def replay_pkl(pkl_path: str, top: int = 1, duration: float = 12.0,
               real_time: bool = True):
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    if "hof" in results:
        # ── GA result ──────────────────────────────────────────────────────
        hof = results["hof"]
        n   = min(top, len(hof))
        for rank, ind in enumerate(hof[:n]):
            fitness = ind.fitness.values[0]
            print(f"\n{'=' * 52}")
            print(f"Hall-of-Fame rank {rank + 1}  |  fitness = {fitness:.3f} m")
            print(f"{'=' * 52}")
            replay_genome(list(ind), duration=duration, real_time=real_time)

    elif "policy_state" in results:
        # ── PG result ──────────────────────────────────────────────────────
        iteration = results.get("iteration", "?")
        print(f"\nLoaded PG policy from iteration {iteration}.")
        policy = GaussianPolicy(OBS_DIM, ACT_DIM)
        policy.load_state_dict(results["policy_state"])
        policy.eval()

        # Print a quick learning-curve summary
        history = results.get("history", [])
        if history:
            last = history[-1]
            print(f"Training summary (last iter):  "
                  f"mean_return={last['mean_return']:.3f}  "
                  f"max_return={last['max_return']:.3f}")

        replay_policy(policy, duration=duration, real_time=real_time)

    else:
        raise ValueError(
            f"'{pkl_path}' does not look like a result from evolve.py or train_pg.py."
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Replay GA or PG results in the MuJoCo interactive viewer"
    )
    p.add_argument("--pkl",      default="results.pkl",
                   help="Result file from evolve.py or train_pg.py "
                        "(default: results.pkl)")
    p.add_argument("--top",      type=int, default=1,
                   help="[GA only] Replay top-N hall-of-fame entries (default: 1)")
    p.add_argument("--duration", type=float, default=12.0,
                   help="Replay duration in seconds (default: 12)")
    p.add_argument("--no-realtime", action="store_true",
                   help="Run as fast as possible instead of real-time")
    p.add_argument("--genome",   nargs="+", type=float,
                   help=f"[GA only] Supply {NUM_ACTUATORS * 3} raw gene values directly")
    return p.parse_args()


if __name__ == "__main__":
    args  = _parse_args()
    rt    = not args.no_realtime

    if args.genome:
        if len(args.genome) != NUM_ACTUATORS * 3:
            raise ValueError(
                f"Expected {NUM_ACTUATORS * 3} gene values, got {len(args.genome)}"
            )
        replay_genome(args.genome, duration=args.duration, real_time=rt)
    else:
        pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.pkl)
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"'{pkl_path}' not found.  Run evolve.py or train_pg.py first."
            )
        replay_pkl(pkl_path, top=args.top, duration=args.duration, real_time=rt)
