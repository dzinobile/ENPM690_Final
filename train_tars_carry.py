"""
SB3 PPO for tars_fused.xml locomotion.

Usage
-----
  python train_tars_carry.py                        # fresh run, 4 parallel envs
  python train_tars_carry.py --timesteps 5000000    # longer run
  python train_tars_carry.py --n-envs 8             # more parallel envs
  python train_tars_carry.py --resume checkpoints/tars_ppo_1000000_steps
  python replay_tars_carry.py --model best_model/best_model  # visualise
"""

import os
import argparse
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR     = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(_DIR, "tars_with_barrel.xml")

# ── Simulation parameters ─────────────────────────────────────────────────────
# 10 physics steps per RL step → action frequency = 1 / (10 * 0.001) = 100 Hz
FRAME_SKIP        = 10
MAX_EPISODE_STEPS = 1000   # 1000 * 10ms = 10 seconds per episode

# ── Joint names in actuator order (must match <actuator> block in XML) ────────
JOINT_NAMES = ["ud_lm", "ud_mm", "ud_rm", "r_lm", "r_mm", "r_rm", "r_lu", "r_ru", "r_ll", "r_rl"]
NUM_JOINTS  = len(JOINT_NAMES)

# ── Reward weights ────────────────────────────────────────────────────────────
W_FORWARD = 3.0    # encourage +x velocity
W_UPRIGHT = 3.0    # penalise tipping (R_zz of torso quaternion)
W_BARREL = 2.0      # encourage barrel parallel to floor
W_HEALTHY = 0.05    # small bonus each step for staying alive
W_ENERGY  = 0.0001  # penalise |torque * joint_vel|
W_ACTION  = 0.0001  # penalise large actions (smooth control)

# ── Termination ───────────────────────────────────────────────────────────────
# MIN_TORSO_Z = 0.20   # fall termination if torso drops below this height
MIN_TORSO_PITCH = -0.8
MAX_TORSO_PITCH = 0.8    # fall termination if torso pitches beyond these angles
MAX_BARREL_HEIGHT = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# Gymnasium environment
# ─────────────────────────────────────────────────────────────────────────────

class TarsEnv(gym.Env):
    """
    MuJoCo environment for TARS (tars_with_barrel.xml) locomotion.

    Observation (41-dim):
        torso z height     (1)
        torso quaternion   (4)   w, x, y, z
        joint positions    (10)   actuator order
        torso lin_vel      (3)
        torso ang_vel      (3)
        joint velocities   (10)   actuator order
        previous action    (10)
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._viewer     = None
        self._step_count = 0

        self._barrel_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "barrel_geom")
        self._barrel_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "barrel")

        self._floor_geom_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        self._mr_geom_box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "middle_geom_box")
        self._mr_geom_cap1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "middle_geom_cap_pos")
        self._mr_geom_cap2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "middle_geom_cap_neg")
        self._mr_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "middle_right")

        self._ml_geom_box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ml_geom_box")
        self._ml_geom_cap1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ml_geom_cap_pos")
        self._ml_geom_cap2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ml_geom_cap_neg")
        self._ml_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "middle_left")
       




        # Pre-compute qpos / qvel addresses in actuator order to avoid the
        # kinematic-tree vs actuator-list mismatch (r_ll and r_ru are swapped).
        self._jnt_qposadr = np.array([
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            ] for n in JOINT_NAMES
        ])
        self._jnt_dofadr = np.array([
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            ] for n in JOINT_NAMES
        ])

        # Read control limits directly from the compiled model (already in SI
        # units: radians for hinges, metres for slides).
        ctrl_lo = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        ctrl_hi = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self._ctrl_mid  = (ctrl_lo + ctrl_hi) / 2.0
        self._ctrl_half = (ctrl_hi - ctrl_lo) / 2.0

        obs_dim = 1 + 4 + NUM_JOINTS + 3 + 3 + NUM_JOINTS + NUM_JOINTS  # = 41

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(NUM_JOINTS,), dtype=np.float32
        )
        self._prev_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._start_y: float = 0.0
        self._barrel_rzz_history: list[float] = []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _joint_qpos(self) -> np.ndarray:
        return self.data.qpos[self._jnt_qposadr].astype(np.float32)

    def _joint_qvel(self) -> np.ndarray:
        return self.data.qvel[self._jnt_dofadr].astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self.data.qpos[2:3],     # torso z height
            self.data.qpos[3:7],     # torso quaternion (w, x, y, z)
            self._joint_qpos(),      # joint positions  (actuator order)
            self.data.qvel[0:3],     # torso linear  velocity
            self.data.qvel[3:6],     # torso angular velocity
            self._joint_qvel(),      # joint velocities (actuator order)
            self._prev_action,       # previous action
        ]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray, torques: np.ndarray, rzz: float) -> float:
        # Forward velocity: -y direction (towards camera)
        r_forward = W_FORWARD * float(-self.data.qvel[1])

        # Upright: quaternion x and y components 
        qx = float(self.data.qpos[4])
        qy = float(self.data.qpos[5])
        r_upright = W_UPRIGHT * (1.0 - 2.0 * (qx**2 + qy**2))

        # rzz is the world-Z component of the barrel's local Z-axis (precomputed in step).
        # When the barrel lies flat (local-Z horizontal), rzz ≈ 0  → reward = 1.
        # When it stands upright (local-Z vertical),      rzz ≈ ±1 → reward = 0.

        b_parallel = W_BARREL * (1.0 - rzz**2)  # barrel parallel to floor
        # Alive bonus: reward for not falling
        r_healthy = W_HEALTHY

        # Energy: penalise mechanical power |τ · q̇|
        r_energy = -W_ENERGY * float(np.abs(torques * self._joint_qvel()).sum())

        # Action smoothness penalty
        r_action = -W_ACTION * float(np.dot(action, action))

        return r_forward + b_parallel + r_healthy + r_energy + r_action + r_upright

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0) # Initialize holding barrel in arms
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:] = [0, 0, 0, 0, 0, 0, 320, 320, 200, 200]
        self._prev_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._step_count  = 0
        self._start_y = float(self.data.qpos[1])
        self._barrel_rzz_history = []
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Map [-1, 1] → full actuator control range (position targets in SI units)
        self.data.ctrl[:] = self._ctrl_mid + action * self._ctrl_half

        # Advance simulation by FRAME_SKIP physics steps
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)

        torques = self.data.actuator_force.copy()
        rzz = float(self.data.xmat[self._barrel_body_id, 8])
        self._barrel_rzz_history.append(rzz)

        obs    = self._get_obs()
        reward = self._compute_reward(action, torques, rzz)

        self._step_count += 1
        w = self.data.qpos[3]
        qx = self.data.qpos[4]
        qy = self.data.qpos[5]
        qz = self.data.qpos[6]
        barrel_z = self.data.xpos[self._barrel_body_id, 2]
        barrel_too_high = barrel_z > MAX_BARREL_HEIGHT

        barrel_on_floor = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            geoms = {c.geom1, c.geom2}
            if self._barrel_geom_id in geoms and self._floor_geom_id in geoms:
                barrel_on_floor = True
                break

        barrel_contact_middle_right = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            geoms = {c.geom1, c.geom2}
            if self._barrel_geom_id in geoms:
                if self._mr_geom_box_id in geoms:
                    barrel_contact_middle_right = True
                    break
                elif self._mr_geom_cap1_id in geoms:
                    barrel_contact_middle_right = True
                    break
                elif self._mr_geom_cap2_id in geoms:
                    barrel_contact_middle_right = True
                    break
                elif self._ml_geom_box_id in geoms:
                    barrel_contact_middle_right = True
                    break
                elif self._ml_geom_cap1_id in geoms:
                    barrel_contact_middle_right = True
                    break
                elif self._ml_geom_cap2_id in geoms:
                    barrel_contact_middle_right = True
                    break

        pitch = np.arcsin(2.0 * (w * qy - qz * qx))
        fallen     = bool(pitch < MIN_TORSO_PITCH or pitch > MAX_TORSO_PITCH)
        terminated = fallen or barrel_too_high or barrel_on_floor #or barrel_contact_middle_right
        truncated  = self._step_count >= MAX_EPISODE_STEPS

        info = {}
        if terminated or truncated:
            parallel = np.array([1.0 - r**2 for r in self._barrel_rzz_history])
            info["distance_traveled"]    = self._start_y - float(self.data.qpos[1])
            info["barrel_parallel_mean"] = float(np.mean(parallel))
            info["barrel_parallel_std"]  = float(np.std(parallel))

        self._prev_action = action
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeStatsCallback(BaseCallback):
    """Logs per-episode distance_traveled and barrel steadiness to TensorBoard."""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "distance_traveled" in info:
                self.logger.record_mean("episode/distance_traveled",    info["distance_traveled"])
                self.logger.record_mean("episode/barrel_parallel_mean", info["barrel_parallel_mean"])
                self.logger.record_mean("episode/barrel_parallel_std",  info["barrel_parallel_std"])
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(timesteps: int, n_envs: int, resume: str | None):
    vec_env  = make_vec_env(TarsEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    eval_env = TarsEnv()

    callbacks = [
        EpisodeStatsCallback(),
        CheckpointCallback(
            save_freq=max(50_000 // n_envs, 1),
            save_path="./checkpoints/",
            name_prefix="tars_carry_ppo",
        ),
        EvalCallback(
            eval_env,
            eval_freq=max(25_000 // n_envs, 1),
            n_eval_episodes=5,
            best_model_save_path="./best_model_carry/",
            verbose=1,
        ),
    ]

    if resume:
        print(f"Resuming from '{resume}'")
        model = PPO.load(resume, env=vec_env)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log="./logs/",
            # Rollout / update
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            # Discount / advantage
            gamma=0.99,
            gae_lambda=0.95,
            # Clipping
            clip_range=0.2,
            # Entropy encourages exploration of joint motions
            ent_coef=0.01,
            # Learning rate
            learning_rate=3e-4,
            # Network: two hidden layers, tanh activation suits [-1, 1] outputs
            policy_kwargs=dict(
                net_arch=[256, 256],
                activation_fn=__import__("torch.nn", fromlist=["Tanh"]).Tanh,
            ),
        )

    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        reset_num_timesteps=(resume is None),
    )
    model.save("tars_carry_ppo_final")
    print("Training complete — model saved to tars_carry_ppo_final.zip")
    vec_env.close()
    eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SB3 PPO for TARS locomotion")
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                        help="Total env steps (default: 2 M)")
    parser.add_argument("--n-envs",   type=int, default=4,
                        help="Parallel envs (default: 4)")
    parser.add_argument("--resume",   type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(args.timesteps, args.n_envs, args.resume)