"""
PPO for MuJoCo quadruped locomotion.

Design follows the referenced paper:
  - Policy:   3-layer MLP [400, 200, 100] with ELU activations
  - Actions:  joint position commands, tracked by an internal PD controller
              (bypasses direct torque control, as in the paper)
  - Obs:      joint positions, lin/ang velocities, previous action  (35-dim)
  - Reward:   forward velocity + upright bonus
              - energy cost - action magnitude - joint limit violations
  - PPO-Clip with a shared Adam optimizer for actor + critic
  - KL-divergence adaptive learning rate (reduce on high KL, raise on low KL)
  - Slightly elevated entropy bonus to promote exploration

Usage
-----
  python train_ppo.py                          # defaults
  python train_ppo.py --iters 500 --steps 2048 # larger run
  python train_ppo.py --resume ppo_tars_checkpoint.pkl
  python replay.py --pkl ppo_tars_results.pkl       # visualise
"""

import math
import os
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import mujoco

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR     = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(_DIR, "tars.xml")

# ── Simulation ────────────────────────────────────────────────────────────────
DT            = 0.001          # must match <option timestep> in tars.xml
MIN_TORSO_Z   = 0.20           # below this → episode ends (fallen)
NUM_ACTUATORS = 10

# ── Joint limits (degrees → radians) ─────────────────────────────────────────

JOINT_LIMITS = np.array([
    [-50, 50], # up-down left-middle
    [-50, 50], # up-down middle-middle
    [-50, 50], # up-down right-middle
    [-355, 355], # rotate left-middle
    [-355, 355], # rotate middle-middle
    [-355, 355], # rotate right-middle
    [-5, 355], # rotate left-arm-upper
    [-5, 355], # rotate right-arm-upper
    [-5, 355], # rotate left-arm-lower
    [-5, 355]  # rotate right-lower

], dtype=np.float32)

for i in range(3,10):
    JOINT_LIMITS[i] = JOINT_LIMITS[i] * (math.pi / 180.0)

# Joint names in actuator order (matches JOINT_LIMITS and data.ctrl order)
JOINT_NAMES = [
    "ud_lm", "ud_mm", "ud_rm",
    "r_lm",  "r_mm",  "r_rm",
    "r_lu",  "r_ru",  "r_ll",  "r_rl",
]

_JT_LO   = JOINT_LIMITS[:, 0]
_JT_HI   = JOINT_LIMITS[:, 1]
_JT_MID  = (_JT_LO + _JT_HI) / 2.0
_JT_HALF = (_JT_HI - _JT_LO) / 2.0


def _joint_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return joint positions in actuator order (matches data.ctrl)."""
    return np.array([data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]]
                     for n in JOINT_NAMES], dtype=np.float32)


def _joint_qvel(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return joint velocities in actuator order (matches data.ctrl)."""
    return np.array([data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]]
                     for n in JOINT_NAMES], dtype=np.float32)

# ── Dimensions ────────────────────────────────────────────────────────────────
# Observation: z(1) + quat(4) + joint_pos(10) + lin_vel(3) + ang_vel(3)
#              + joint_vel(10) + prev_action(10) = 41
OBS_DIM = 41
ACT_DIM = NUM_ACTUATORS

# ── Reward weights ─────────────────────────────────────────────────────────────
W_FORWARD     = 5.0
W_UPRIGHT     = 1.0
W_ENERGY      = 0.0
W_ACTION      = 0.0
W_JOINT_LIMIT = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Observation and control helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_obs(model: mujoco.MjModel, data: mujoco.MjData, prev_action: np.ndarray) -> np.ndarray:
    """Build a 41-dim observation vector from MjData + previous action."""
    return np.concatenate([
        data.qpos[2:3],              # torso z height
        data.qpos[3:7],              # torso quaternion  (w, x, y, z)
        _joint_qpos(model, data),    # 10 joint positions in actuator order
        data.qvel[0:3],              # torso linear  velocity
        data.qvel[3:6],              # torso angular velocity
        _joint_qvel(model, data),    # 10 joint velocities in actuator order
        prev_action,                 # previous action   (for smoothness signal)
    ]).astype(np.float32)


def action_to_target(action: np.ndarray) -> np.ndarray:
    """
    Map tanh-space actions in [-1, 1] → target joint angles within limits.
    Uses mid-point + half-range scaling so the full [-1, 1] maps to the
    full joint range with the neutral pose at 0.
    """
    return _JT_MID + action * _JT_HALF


def apply_pd_control(data: mujoco.MjData, action: np.ndarray) -> np.ndarray:
    """
    Set position actuator targets. TARS uses <position> actuators so ctrl is
    the target joint position directly — MuJoCo handles the internal PD.
    Returns the actuator forces (after the step) for use in the energy reward.
    """
    data.ctrl[:] = action_to_target(action)
    return data.actuator_force.copy()


def compute_reward(model: mujoco.MjModel, data: mujoco.MjData,
                   action: np.ndarray, torques: np.ndarray) -> float:
    """
    r = forward_velocity
      + upright_bonus
      - energy_cost        (mechanical power |τ · dq|)
      - action_magnitude
      - joint_limit_penalty
    """
    # Forward velocity
    r_forward = W_FORWARD * float(data.qvel[0])

    # Upright bonus — how aligned is the torso's local z-axis with world z?
    # R_zz = 1 - 2(qx² + qy²)  from the rotation-matrix quaternion formula
    qx, qy = float(data.qpos[4]), float(data.qpos[5])
    upright   = 1.0 - 2.0 * (qx**2 + qy**2)       # ∈ [-1, 1]
    r_upright = W_UPRIGHT * upright

    # Energy cost (mechanical power: actuator force × joint velocity)
    joint_vel = _joint_qvel(model, data)
    r_energy  = -W_ENERGY * float(np.abs(torques * joint_vel).sum())

    # Action magnitude penalty
    r_action  = -W_ACTION * float(np.dot(action, action))

    # Joint limit violation penalty
    q_joints   = _joint_qpos(model, data)
    violations = (np.maximum(0.0, _JT_LO - q_joints)
                + np.maximum(0.0, q_joints - _JT_HI))
    r_joint    = -W_JOINT_LIMIT * float(violations.sum())

    return r_forward + r_upright + r_energy + r_action + r_joint


# ─────────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: tuple, out_dim: int,
         act_cls=nn.ELU) -> nn.Sequential:
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), act_cls()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Gaussian policy with a 3-layer ELU MLP mean network  [400, 200, 100]
    and a separate learnable log-std parameter vector.
    """

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM,
                 hidden=(400, 200, 100), log_std_init=-0.5):
        super().__init__()
        self.net     = _mlp(obs_dim, hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))

    def distribution(self, obs: torch.Tensor) -> Normal:
        return Normal(self.net(obs), self.log_std.exp().expand_as(self.net(obs)))

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, deterministic: bool = False):
        """
        Sample one action.
        Returns (action_np clipped to [-1,1], log_prob scalar tensor).
        """
        obs  = torch.from_numpy(obs_np).unsqueeze(0)
        dist = self.distribution(obs)
        a    = dist.mean if deterministic else dist.sample()
        lp   = dist.log_prob(a).sum(-1).squeeze(0)
        return a.squeeze(0).clamp(-1.0, 1.0).numpy(), lp

    def evaluate(self, obs: torch.Tensor, acts: torch.Tensor):
        """Return (log_probs, entropy) for a batch — used during PPO update."""
        dist    = self.distribution(obs)
        log_probs = dist.log_prob(acts).sum(-1)
        entropy   = dist.entropy().sum(-1)
        return log_probs, entropy


class Critic(nn.Module):
    """State-value function V(s), same MLP architecture as the actor."""

    def __init__(self, obs_dim=OBS_DIM, hidden=(400, 200, 100)):
        super().__init__()
        self.net = _mlp(obs_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout buffer
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Fixed-length buffer that stores transitions across episode boundaries.
    After collection, call compute_gae() to fill advantages and returns.
    """

    def __init__(self, n_steps: int, obs_dim: int = OBS_DIM,
                 act_dim: int = ACT_DIM):
        self.n_steps = n_steps
        self.obs       = np.zeros((n_steps, obs_dim),  dtype=np.float32)
        self.acts      = np.zeros((n_steps, act_dim),  dtype=np.float32)
        self.log_probs = np.zeros(n_steps,             dtype=np.float32)
        self.rewards   = np.zeros(n_steps,             dtype=np.float32)
        self.values    = np.zeros(n_steps,             dtype=np.float32)
        self.dones     = np.zeros(n_steps,             dtype=np.float32)
        self.advantages = None
        self.returns    = None
        self._ptr = 0

    def add(self, obs, act, log_prob, reward, value, done):
        self.obs[self._ptr]       = obs
        self.acts[self._ptr]      = act
        self.log_probs[self._ptr] = log_prob
        self.rewards[self._ptr]   = reward
        self.values[self._ptr]    = value
        self.dones[self._ptr]     = done
        self._ptr += 1

    def full(self) -> bool:
        return self._ptr >= self.n_steps

    def reset(self):
        self._ptr = 0

    def compute_gae(self, last_value: float, gamma: float = 0.99,
                    gae_lambda: float = 0.95):
        """
        Generalised Advantage Estimation (Schulman et al., 2016).

        δ_t   = r_t + γ V(s_{t+1}) (1−done) − V(s_t)
        Â_t  = Σ_{k≥0}  (γλ)^k  δ_{t+k}
        targets = Â + V  (TD-λ returns for value loss)
        """
        advantages = np.zeros(self.n_steps, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            not_done   = 1.0 - self.dones[t]
            next_val   = last_value if t == self.n_steps - 1 else self.values[t + 1]
            delta      = self.rewards[t] + gamma * next_val * not_done - self.values[t]
            gae        = delta + gamma * gae_lambda * not_done * gae
            advantages[t] = gae
        self.advantages = advantages
        self.returns    = advantages + self.values

    def tensors(self):
        return (
            torch.from_numpy(self.obs),
            torch.from_numpy(self.acts),
            torch.from_numpy(self.log_probs),
            torch.from_numpy(self.advantages),
            torch.from_numpy(self.returns),
        )


# ─────────────────────────────────────────────────────────────────────────────
# KL-adaptive learning rate scheduler
# ─────────────────────────────────────────────────────────────────────────────

class KLAdaptiveLR:
    """
    After each PPO update, compare measured KL against a target.
    If KL is too high  → policy changed too much → lower LR.
    If KL is too low   → policy barely moved    → raise LR.
    """

    def __init__(self, optimizer, target_kl: float = 0.01,
                 factor: float = 1.5,
                 min_lr: float = 1e-4, max_lr: float = 1e-2):
        self.optimizer  = optimizer
        self.target_kl  = target_kl
        self.factor     = factor
        self.min_lr     = min_lr
        self.max_lr     = max_lr

    def step(self, measured_kl: float) -> float:
        for pg in self.optimizer.param_groups:
            if measured_kl > self.target_kl * 1.5:
                pg["lr"] = max(self.min_lr, pg["lr"] / self.factor)
            elif measured_kl < self.target_kl / 1.5:
                pg["lr"] = min(self.max_lr, pg["lr"] * self.factor)
        return self.optimizer.param_groups[0]["lr"]


# ─────────────────────────────────────────────────────────────────────────────
# Rollout collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_rollout(actor, critic, model, data, buffer,
                    env_obs, env_done, env_prev_act):
    """
    Fill `buffer` with n_steps transitions, resetting the environment at
    episode boundaries.  Returns (next_obs, next_done, next_prev_act,
    bootstrap_value) for GAE.
    """
    buffer.reset()
    obs      = env_obs
    done     = env_done
    prev_act = env_prev_act

    while not buffer.full():
        if done:
            mujoco.mj_resetData(model, data)
            prev_act = np.zeros(ACT_DIM, dtype=np.float32)
            obs = get_obs(model, data, prev_act)
            done = False

        act, log_prob = actor.act(obs)

        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            value = critic(obs_t).item()

        torques = apply_pd_control(data, act)
        mujoco.mj_step(model, data)

        reward  = compute_reward(model, data, act, torques)
        done    = bool(data.qpos[2] < MIN_TORSO_Z)
        next_obs = get_obs(model, data, act)

        buffer.add(obs, act, log_prob.item(), reward, value, float(done))
        obs, prev_act = next_obs, act

    # Bootstrap value for the last state
    with torch.no_grad():
        last_val = 0.0 if done else critic(
            torch.from_numpy(obs).unsqueeze(0)
        ).item()

    return obs, done, prev_act, last_val


# ─────────────────────────────────────────────────────────────────────────────
# PPO update
# ─────────────────────────────────────────────────────────────────────────────

def ppo_update(actor, critic, actor_optimizer, critic_optimizer, buffer,
               n_epochs=10, batch_size=64, clip_eps=0.2,
               entropy_coef=0.005, max_grad_norm=0.5) -> dict:
    """
    Run K epochs of mini-batch PPO updates over the collected buffer.

    Actor and critic are updated with separate optimizers so that the
    value-function MSE (which operates on large-scale returns) cannot
    bleed into the actor's gradient through shared optimizer momentum.
    Returns diagnostic scalars.
    """
    obs_b, acts_b, old_lp_b, adv_b, ret_b = buffer.tensors()

    # Normalise advantages across the full batch
    adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

    n_samples   = len(obs_b)
    total_loss  = total_kl = 0.0
    n_updates   = 0

    for _ in range(n_epochs):
        perm = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]

            new_lp, entropy = actor.evaluate(obs_b[idx], acts_b[idx])

            # ── Actor loss (PPO-clip + entropy bonus) ──────────────────────
            ratio  = (new_lp - old_lp_b[idx]).exp()
            surr1  = ratio * adv_b[idx]
            surr2  = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv_b[idx]
            actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy.mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=max_grad_norm)
            actor_optimizer.step()

            # ── Critic loss (MSE, separate backward pass) ──────────────────
            values     = critic(obs_b[idx])
            value_loss = 0.5 * (values - ret_b[idx]).pow(2).mean()

            critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=max_grad_norm)
            critic_optimizer.step()

            with torch.no_grad():
                # Divide by ACT_DIM so KL is per-dimension and comparable to
                # single-action environments.  Without this, an 8-action policy
                # reports KL ~8× too large, crashing the adaptive LR scheduler.
                kl = (old_lp_b[idx] - new_lp).mean().item() / ACT_DIM

            total_loss += actor_loss.item()
            total_kl   += kl
            n_updates  += 1

    return {"loss": total_loss / n_updates, "kl": total_kl / n_updates}


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    n_iterations:     int   = 500,
    n_steps:          int   = 2048,   # rollout steps per update
    n_epochs:         int   = 10,     # PPO update epochs per rollout
    batch_size:       int   = 64,
    actor_lr:         float = 3e-4,   # adaptive via KL scheduler
    critic_lr:        float = 1e-3,   # fixed; higher OK since loss is isolated
    gamma:            float = 0.99,
    gae_lambda:       float = 0.95,
    clip_eps:         float = 0.2,
    entropy_coef:     float = 0.005,  # reduced: prevents std from growing
    target_kl:        float = 0.01,   # per-dimension KL target
    max_grad_norm:    float = 0.5,
    checkpoint_every: int   = 25,
    results_path:     str   = "ppo_tars_results.pkl",
    resume_path:      str   = None,
):
    model  = mujoco.MjModel.from_xml_path(XML_PATH)
    data   = mujoco.MjData(model)

    actor    = Actor()
    critic   = Critic()
    # Separate optimizers: critic LR is fixed (its MSE scale doesn't affect actor)
    actor_optimizer  = optim.Adam(actor.parameters(),  lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
    kl_sched  = KLAdaptiveLR(actor_optimizer, target_kl=target_kl)
    buffer    = RolloutBuffer(n_steps)
    history   = []
    start_it  = 0

    if resume_path and os.path.exists(resume_path):
        ckpt = _load(resume_path)
        actor.load_state_dict(ckpt["actor_state"])
        critic.load_state_dict(ckpt["critic_state"])
        actor_optimizer.load_state_dict(ckpt["actor_optimizer_state"])
        critic_optimizer.load_state_dict(ckpt["critic_optimizer_state"])
        history  = ckpt.get("history", [])
        start_it = ckpt.get("iteration", 0)
        print(f"Resumed from '{resume_path}' at iteration {start_it}.")

    # Initialise environment state
    mujoco.mj_resetData(model, data)
    env_prev_act = np.zeros(ACT_DIM, dtype=np.float32)
    env_obs      = get_obs(model, data, env_prev_act)
    env_done     = False

    hdr = (f"{'Iter':>5}  {'MeanRew':>9}  {'TotalRew':>9}  "
           f"{'KL':>8}  {'Loss':>9}  {'LR':>8}  {'Std':>7}")
    print(hdr)
    print("-" * len(hdr))

    for it in range(start_it, start_it + n_iterations):

        env_obs, env_done, env_prev_act, last_val = collect_rollout(
            actor, critic, model, data, buffer,
            env_obs, env_done, env_prev_act,
        )

        buffer.compute_gae(last_val, gamma, gae_lambda)

        diag   = ppo_update(actor, critic, actor_optimizer, critic_optimizer,
                            buffer, n_epochs, batch_size, clip_eps,
                            entropy_coef, max_grad_norm)
        new_lr = kl_sched.step(diag["kl"])

        stats = {
            "iteration":   it + 1,
            "mean_reward": float(buffer.rewards.mean()),
            "total_reward": float(buffer.rewards.sum()),
            "kl":          diag["kl"],
            "loss":        diag["loss"],
            "lr":          new_lr,
            "mean_std":    actor.log_std.exp().mean().item(),
        }
        history.append(stats)
        print(f"{it+1:>5}  {stats['mean_reward']:>9.4f}  "
              f"{stats['total_reward']:>9.2f}  "
              f"{stats['kl']:>8.5f}  {stats['loss']:>9.5f}  "
              f"{new_lr:>8.2e}  {stats['mean_std']:>7.4f}")

        if (it + 1) % checkpoint_every == 0:
            _save("ppo_tars_checkpoint.pkl", actor, critic,
                  actor_optimizer, critic_optimizer, history, it + 1)
            print(f"  [checkpoint → ppo_tars_checkpoint.pkl]")

    _save(results_path, actor, critic, actor_optimizer, critic_optimizer,
          history, start_it + n_iterations)
    print(f"\nTraining complete.  Results saved to '{results_path}'.")
    return actor, critic, history


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def _save(path, actor, critic, actor_optimizer, critic_optimizer,
          history, iteration):
    with open(path, "wb") as f:
        pickle.dump({
            "actor_state":          actor.state_dict(),
            "critic_state":         critic.state_dict(),
            "actor_optimizer_state":  actor_optimizer.state_dict(),
            "critic_optimizer_state": critic_optimizer.state_dict(),
            "history":              history,
            "iteration":            iteration,
        }, f)


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="PPO for MuJoCo quadruped locomotion"
    )
    p.add_argument("--iters",         type=int,   default=500,   help="Gradient updates (default: 500)")
    p.add_argument("--steps",         type=int,   default=2048,  help="Rollout steps per update (default: 2048)")
    p.add_argument("--epochs",        type=int,   default=10,    help="PPO update epochs (default: 10)")
    p.add_argument("--batch",         type=int,   default=64,    help="Mini-batch size (default: 64)")
    p.add_argument("--actor-lr",      type=float, default=3e-4,  help="Actor initial LR, adaptive (default: 3e-4)")
    p.add_argument("--critic-lr",     type=float, default=1e-3,  help="Critic fixed LR (default: 1e-3)")
    p.add_argument("--clip",          type=float, default=0.2,   help="PPO clip epsilon (default: 0.2)")
    p.add_argument("--entropy",       type=float, default=0.005, help="Entropy coefficient (default: 0.005)")
    p.add_argument("--target-kl",     type=float, default=0.01,  help="Per-dim KL target for adaptive LR (default: 0.01)")
    p.add_argument("--ckpt-every",    type=int,   default=25,    help="Checkpoint interval (default: 25)")
    p.add_argument("--resume",        type=str,   default=None,  help="Resume from checkpoint")
    p.add_argument("--out",           type=str,   default="ppo_tars_results.pkl", help="Output file")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        n_iterations     = args.iters,
        n_steps          = args.steps,
        n_epochs         = args.epochs,
        batch_size       = args.batch,
        actor_lr         = args.actor_lr,
        critic_lr        = args.critic_lr,
        clip_eps         = args.clip,
        entropy_coef     = args.entropy,
        target_kl        = args.target_kl,
        checkpoint_every = args.ckpt_every,
        results_path     = args.out,
        resume_path      = args.resume,
    )
