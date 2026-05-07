"""
Replay a sequence of checkpoints in order so you can watch training progress.

Edit CHECKPOINTS below to choose which ones to show.
Each checkpoint runs for EPISODES_PER_CHECKPOINT episodes, then the next loads.
Press Ctrl+C or close the viewer to skip to the next checkpoint.

Usage
-----
  python replay_checkpoints.py
  python replay_checkpoints.py --keyframe 0
  python replay_checkpoints.py --env carry          # carry (default) or walk
"""

import argparse
import subprocess
import time
import numpy as np
import mujoco
from stable_baselines3 import PPO
import os

# ── Edit this list to choose which checkpoints to show ───────────────────────
CHECKPOINTS = [
    "checkpoints/carry/tars_carry_ppo_49998_steps.zip",
    "checkpoints/carry/tars_carry_ppo_999960_steps.zip",
    "checkpoints/carry/tars_carry_ppo_2000000_steps.zip",
    "checkpoints/carry/tars_carry_ppo_3000000_steps.zip",  
    "checkpoints/carry/tars_carry_ppo_4049838_steps.zip",
    "checkpoints/carry/tars_carry_ppo_5049798_steps.zip",  
    "best_model_carry/best_model.zip",
]

EPISODES_PER_CHECKPOINT = 1
# ─────────────────────────────────────────────────────────────────────────────


def make_env(env_name: str, keyframe: int | None):
    if env_name == "carry":
        from train_tars_carry import TarsEnv
    else:
        from train_tars_walk import TarsEnv

    import mujoco as _mj

    env = TarsEnv(render_mode="human")

    if keyframe is not None:
        def _fixed_reset(seed=None, options=None):
            super(type(env), env).reset(seed=seed)
            _mj.mj_resetDataKeyframe(env.model, env.data, keyframe)
            _mj.mj_forward(env.model, env.data)
            env._prev_action[:] = 0
            env._step_count = 0
            env._start_y = float(env.data.xpos[env._base_link_id, 1])
            if hasattr(env, "_human_rzz_history"):
                env._human_rzz_history = []
            return env._get_obs(), {}
        env.reset = _fixed_reset

    return env


def _configure_viewer(env):
    """Set tracking camera and fullscreen. Called once after viewer is created."""
    v = env._viewer
    v.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
    v.cam.trackbodyid = env._base_link_id
    v.cam.distance    = 5.0
    v.cam.elevation   = -20.0
    v.cam.azimuth     = 90.0
    # Best-effort fullscreen on Linux via wmctrl
    time.sleep(0.2)  # give the window time to appear
    try:
        subprocess.run(
            ["wmctrl", "-r", "MuJoCo", "-b", "add,fullscreen"],
            capture_output=True, timeout=1,
        )
    except Exception:
        pass


def run_checkpoint(model_path: str, env, n_episodes: int) -> bool:
    """Run n_episodes with model_path. Returns False if viewer was closed."""
    print(f"\n{'─'*60}")
    print(f"  Checkpoint: {model_path}")
    print(f"{'─'*60}")

    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"  [SKIP] File not found: {model_path}.zip")
        return True

    viewer_ready = False
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        step = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
            env.render()

            if not viewer_ready and env._viewer is not None:
                _configure_viewer(env)
                viewer_ready = True

            if env._viewer is not None and not env._viewer.is_running():
                return False

            time.sleep(0.01)

            if terminated or truncated:
                reason = "fell" if terminated else "timeout"
                print(f"  ep {ep}/{n_episodes} | steps {step:4d} | "
                      f"reward {total_reward:8.2f} | {reason}")
                break

        if env._viewer is not None and not env._viewer.is_running():
            return False

    return True


def main(env_name: str, keyframe: int | None):
    env = make_env(env_name, keyframe)

    for ckpt in CHECKPOINTS:
        keep_going = run_checkpoint(ckpt, env, EPISODES_PER_CHECKPOINT)
        if not keep_going:
            print("\nViewer closed — stopping.")
            break

    try:
        env.close()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay checkpoints in sequence")
    parser.add_argument("--env",      type=str, default="carry", choices=["carry", "walk"],
                        help="Which environment to use (default: carry)")
    parser.add_argument("--keyframe", type=int, default=None,
                        help="Pin to a fixed keyframe (0 or 1) for consistent starts")
    args = parser.parse_args()

    main(args.env, args.keyframe)
