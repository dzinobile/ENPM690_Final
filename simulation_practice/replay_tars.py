"""
Replay evolved or trained gaits in MuJoCo's interactive viewer.

Supports all three result formats — auto-detected from the pickle contents:
  - evolve.py    → results.pkl       (GA, contains hall-of-fame)
  - train_pg.py  → pg_results.pkl    (vanilla PG, torque control)
  - train_ppo.py → ppo_tars_results.pkl   (PPO, position / PD control)

Requirements
------------
  A display (X11 / Wayland) is needed.  On a headless server, prefix with:
    MUJOCO_GL=osmesa python replay_tars.py   (software rendering)

Usage
-----
  python replay_tars.py                              # GA best genome
  python replay_tars.py --top 3                      # GA top-3 hall-of-fame
  python replay_tars.py --pkl pg_results.pkl         # vanilla PG policy
  python replay_tars.py --pkl ppo_tars_results.pkl        # PPO policy
  python replay_tars.py --pkl ppo_tars_checkpoint.pkl     # mid-training PPO
  python replay_tars.py --genome 0.82 0.41 1.57 ...  # GA genome from CLI (24 values)
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
# from evolve    import decode, XML_PATH, DT, NUM_ACTUATORS
# from train_pg  import GaussianPolicy as PGPolicy, get_obs as pg_get_obs
# from train_pg  import OBS_DIM as PG_OBS_DIM, ACT_DIM
# from train_ppo import Actor as PPOActor, get_obs as ppo_get_obs
# from train_ppo import OBS_DIM as PPO_OBS_DIM, apply_pd_control
from train_tars_ppo import  XML_PATH, DT, NUM_ACTUATORS
from train_tars_ppo import Actor as PPOActor, get_obs as ppo_get_obs
from train_tars_ppo import OBS_DIM as PPO_OBS_DIM, apply_pd_control, ACT_DIM


# ── Shared viewer loop ────────────────────────────────────────────────────────

def _run_viewer(model, data, steps, real_time, ctrl_fn):
    """
    Core render loop.  ctrl_fn(step) is called each step and may either
    return an iterable of control values (which we set on data.ctrl),
    or None if it has already written data.ctrl itself (e.g. PPO PD path).
    """
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Lock the camera onto the torso so it follows the robot
        viewer.cam.trackbodyid = model.body("middle_right").id
        viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance    = 3.0    # metres from the robot
        viewer.cam.elevation   = -20    # degrees (negative = looking slightly down)
        viewer.cam.azimuth     = 90     # degrees (side-on view along x-axis)

        t0 = time.time()
        for step in range(steps):
            if not viewer.is_running():
                break

            ctrl = ctrl_fn(step)
            if ctrl is not None:
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





# ── PPO replay ────────────────────────────────────────────────────────────────

def replay_ppo(actor: PPOActor, duration: float = 12.0, real_time: bool = True):
    """Replay a PPO policy deterministically (position / PD control)."""
    model    = mujoco.MjModel.from_xml_path(XML_PATH)
    data     = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    steps    = int(duration / DT)
    prev_act = np.zeros(ACT_DIM, dtype=np.float32)

    print(f"\nReplaying PPO policy for {duration} s  (close window to quit)")
    print("Using deterministic mean with PD position control.")

    def ctrl_fn(_step):
        nonlocal prev_act
        obs = torch.from_numpy(ppo_get_obs(model, data, prev_act)).unsqueeze(0)
        with torch.no_grad():
            act = actor.net(obs).squeeze(0).clamp(-1.0, 1.0).numpy()
        apply_pd_control(data, act)   # writes data.ctrl directly
        prev_act = act
        return None                   # ctrl already set

    _run_viewer(model, data, steps, real_time, ctrl_fn)


# ── Auto-detect and dispatch ──────────────────────────────────────────────────

def replay_pkl(pkl_path: str, top: int = 1, duration: float = 12.0,
               real_time: bool = True):
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    iteration = results.get("iteration", "?")


    if "actor_state" in results:
        # ── PPO ───────────────────────────────────────────────────────────
        print(f"\nLoaded PPO actor from iteration {iteration}.")
        actor = PPOActor(PPO_OBS_DIM, ACT_DIM)
        actor.load_state_dict(results["actor_state"])
        actor.eval()
        history = results.get("history", [])
        if history:
            last = history[-1]
            print(f"Last iter:  mean_reward={last['mean_reward']:.4f}  "
                  f"kl={last['kl']:.5f}  lr={last['lr']:.2e}")
        replay_ppo(actor, duration=duration, real_time=real_time)


    else:
        raise ValueError(
            f"'{pkl_path}' is not a recognised result file from this project."
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Replay GA / PG / PPO results in the MuJoCo interactive viewer"
    )
    p.add_argument("--pkl",         default="results.pkl",
                   help="Result file (default: results.pkl)")
    p.add_argument("--top",         type=int,   default=1,
                   help="[GA only] Replay top-N hall-of-fame entries (default: 1)")
    p.add_argument("--duration",    type=float, default=12.0,
                   help="Replay duration in seconds (default: 12)")
    p.add_argument("--no-realtime", action="store_true",
                   help="Run as fast as possible instead of real-time")
    p.add_argument("--genome",      nargs="+",  type=float,
                   help=f"[GA only] {NUM_ACTUATORS * 3} raw gene values from CLI")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rt   = not args.no_realtime


    
    pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            args.pkl)
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"'{pkl_path}' not found.  Run a training script first."
        )
    replay_pkl(pkl_path, top=args.top, duration=args.duration, real_time=rt)
