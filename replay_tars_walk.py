"""
Replay a trained TARS PPO model with MuJoCo viewer.

Usage
-----
  python replay_tars_walk.py                              # uses best_model/best_model
  python replay_tars_walk.py --model tars_ppo_final       # specific checkpoint
  python replay_tars_walk.py --model checkpoints/tars_ppo_500000_steps
  python replay_tars_walk.py --episodes 5                 # run 5 episodes
"""

import argparse
import os
import time
import numpy as np
from stable_baselines3 import PPO
from train_tars_walk import TarsEnv


def replay(model_path: str, n_episodes: int, keyframe: int | None):
    print(f"Loading model from '{model_path}'...")
    model = PPO.load(model_path)

    env = TarsEnv(render_mode="human")

    if keyframe is not None:
        # Monkey-patch reset to use a fixed keyframe so replays are reproducible
        import mujoco as _mj
        _orig_reset = env.reset
        def _fixed_reset(seed=None, options=None):
            super(type(env), env).reset(seed=seed)
            _mj.mj_resetDataKeyframe(env.model, env.data, keyframe)
            _mj.mj_forward(env.model, env.data)
            env._prev_action[:] = 0
            env._step_count = 0
            env._start_y = float(env.data.xpos[env._base_link_id, 1])
            env._human_rzz_history = []
            return env._get_obs(), {}
        env.reset = _fixed_reset

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

            # Stop if viewer window was closed by the user
            if env._viewer is not None and not env._viewer.is_running():
                print("Viewer closed by user.")
                break

            time.sleep(0.01)   # slow down to ~100 fps so it's watchable

            if terminated or truncated:
                reason = "fell" if terminated else "timeout"
                print(f"Episode {ep:3d} | steps {step:5d} | "
                      f"reward {total_reward:8.2f} | {reason}")
                break

        # Stop all episodes if viewer was closed
        if env._viewer is not None and not env._viewer.is_running():
            break

    try:
        env.close()
    except Exception:
        pass

    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay trained TARS PPO agent")
    parser.add_argument("--model",    type=str, default="best_model/best_model",
                        help="Path to saved model (no .zip extension)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to replay (default: 3)")
    parser.add_argument("--keyframe", type=int, default=None,
                        help="Pin to a specific keyframe (0 or 1) for reproducible replays; omit to use random like training")
    args = parser.parse_args()

    replay(args.model, args.episodes, args.keyframe)
