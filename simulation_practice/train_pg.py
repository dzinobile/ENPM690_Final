"""
Vanilla Policy Gradient (REINFORCE) for MuJoCo quadruped locomotion.

Algorithm
---------
REINFORCE with a mean-return baseline and advantage normalisation.

For each iteration:
  1. Collect EPISODES_PER_UPDATE rollouts using the current policy π_θ.
  2. Compute discounted returns G_t at each timestep.
  3. Subtract the mean G_0 (episode return) as a variance-reduction baseline.
  4. Normalise advantages across the batch.
  5. Gradient step:  θ ← θ + α · ∇_θ  Σ_t  log π_θ(aₜ | sₜ) · Âₜ

Policy
------
Gaussian MLP:  obs → μ(obs)  with a separate learnable log σ.
Actions are sampled from N(μ, σ) and clipped to [-1, 1] before being
sent to MuJoCo.  Log-probabilities are computed on the pre-clip sample
(standard REINFORCE convention).

Observation  (27-dim)
---------------------
  torso z (1) | torso quaternion (4) | joint angles (8) | all velocities (14)
  Absolute x/y are excluded so the policy learns a pose-relative gait.

Reward (per timestep)
---------------------
  r_t = v_x  -  0.001 · ||aₜ||²
  (forward velocity minus a small energy penalty)

Usage
-----
  python train_pg.py                         # defaults
  python train_pg.py --iters 300 --eps 16    # bigger run
  python train_pg.py --resume pg_results.pkl # continue training
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
XML_PATH = os.path.join(_DIR, "crawler.xml")

# ── Simulation constants ───────────────────────────────────────────────────────
DT             = 0.005   # must match <option timestep> in crawler.xml
SIM_DURATION   = 5.0     # seconds per episode
NUM_ACTUATORS  = 8
MIN_TORSO_Z    = 0.20    # early-termination height threshold

# ── Observation / action dimensions ───────────────────────────────────────────
# qpos layout: [x, y, z,  qw, qx, qy, qz,  j0..j7]  → 15 values
# qvel layout: [vx, vy, vz,  wx, wy, wz,  dj0..dj7] → 14 values
# We drop absolute x, y from qpos; everything else goes in.
OBS_DIM = 1 + 4 + 8 + 14   # = 27
ACT_DIM = NUM_ACTUATORS     # = 8

# ── Reward hyper-parameters ────────────────────────────────────────────────────
CTRL_COST_WEIGHT = 0.001
GAMMA            = 0.99


# ── Observation helper ────────────────────────────────────────────────────────

def get_obs(data: mujoco.MjData) -> np.ndarray:
    """Extract a 27-dim observation vector from MjData."""
    return np.concatenate([
        data.qpos[2:3],   # torso z
        data.qpos[3:7],   # quaternion  (w, x, y, z)
        data.qpos[7:],    # 8 joint angles
        data.qvel,        # 14 velocities (linear + angular + joint)
    ]).astype(np.float32)


# ── Policy network ─────────────────────────────────────────────────────────────

class GaussianPolicy(nn.Module):
    """
    Two-hidden-layer MLP Gaussian policy.

    Outputs a mean for each action.  A single learnable log_std parameter
    (not state-dependent) is shared across all actions — this is the classic
    REINFORCE setup and keeps the gradient landscape simple.
    """

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        # Start with std ≈ 0.5 to encourage exploration
        self.log_std = nn.Parameter(torch.full((act_dim,), math.log(0.5)))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std  = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray):
        """
        Sample one action and return (action_np, log_prob_tensor).

        The returned action is clipped to [-1, 1]; the log-prob is computed
        on the raw (pre-clip) sample — standard REINFORCE convention.
        """
        obs  = torch.from_numpy(obs_np).unsqueeze(0)
        dist = self.distribution(obs)
        a    = dist.sample()                    # shape (1, ACT_DIM)
        lp   = dist.log_prob(a).sum(-1)         # scalar log-prob
        return a.squeeze(0).clamp(-1.0, 1.0).numpy(), lp.squeeze(0)

    def log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Re-compute log π(a | s) with gradient tape (used during update)."""
        return self.distribution(obs).log_prob(act).sum(-1)


# ── Rollout ───────────────────────────────────────────────────────────────────

def rollout(policy: GaussianPolicy, model: mujoco.MjModel,
            data: mujoco.MjData) -> dict:
    """
    Run one episode and return collected transitions.

    Returns
    -------
    dict with keys:
      obs      : (T, OBS_DIM) float32 array
      acts     : (T, ACT_DIM) float32 array
      rewards  : (T,) float32 array
      returns  : (T,) float32 array  (discounted, from each t to end)
    """
    mujoco.mj_resetData(model, data)

    obs_buf, act_buf, rew_buf = [], [], []
    steps = int(SIM_DURATION / DT)

    for _ in range(steps):
        obs = get_obs(data)
        act, _ = policy.act(obs)               # sample under current policy

        data.ctrl[:] = act
        mujoco.mj_step(model, data)

        reward = float(data.qvel[0]) - CTRL_COST_WEIGHT * float(np.sum(act ** 2))

        obs_buf.append(obs)
        act_buf.append(act)
        rew_buf.append(reward)

        if data.qpos[2] < MIN_TORSO_Z:         # fallen — end early
            break

    # Discounted returns  G_t = Σ_{k≥t} γ^(k-t) r_k
    G, returns = 0.0, []
    for r in reversed(rew_buf):
        G = r + GAMMA * G
        returns.insert(0, G)

    return {
        "obs":     np.array(obs_buf,  dtype=np.float32),
        "acts":    np.array(act_buf,  dtype=np.float32),
        "rewards": np.array(rew_buf,  dtype=np.float32),
        "returns": np.array(returns,  dtype=np.float32),
    }


# ── Policy-gradient update ────────────────────────────────────────────────────

def pg_update(policy: GaussianPolicy, optimizer: optim.Optimizer,
              episodes: list) -> dict:
    """
    One REINFORCE gradient step over a batch of episodes.

    Returns a dict of scalar diagnostics.
    """
    # Concatenate batch
    obs_batch = torch.from_numpy(np.concatenate([e["obs"]     for e in episodes]))
    act_batch = torch.from_numpy(np.concatenate([e["acts"]    for e in episodes]))
    ret_batch = torch.from_numpy(np.concatenate([e["returns"] for e in episodes]))

    # Baseline: subtract mean episode return (G_0) for each episode,
    # broadcast so every timestep in that episode uses its own episode baseline.
    ep_means = torch.tensor(
        [e["returns"][0] for e in episodes], dtype=torch.float32
    )
    baseline = ep_means.mean()

    advantages = ret_batch - baseline

    # Normalise advantages (reduces gradient variance, standard practice)
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss:  -E[ log π(a|s) · Â ]
    log_probs = policy.log_prob(obs_batch, act_batch)   # (T,)
    loss = -(log_probs * advantages).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    optimizer.step()

    return {
        "loss":      loss.item(),
        "baseline":  baseline.item(),
        "mean_std":  policy.log_std.exp().mean().item(),
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    n_iterations:      int   = 200,
    episodes_per_iter: int   = 10,
    lr:                float = 3e-4,
    hidden:            int   = 64,
    checkpoint_every:  int   = 20,
    results_path:      str   = "pg_results.pkl",
    resume_path:       str   = None,
):
    """
    Parameters
    ----------
    n_iterations       : number of gradient updates
    episodes_per_iter  : episodes collected per update (higher = lower variance)
    lr                 : Adam learning rate
    hidden             : hidden layer width
    checkpoint_every   : save a checkpoint every N iterations
    results_path       : file to write final results
    resume_path        : if set, load policy weights and history from this file
    """
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    policy    = GaussianPolicy(OBS_DIM, ACT_DIM, hidden)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    history   = []   # list of per-iteration stat dicts
    start_iter = 0

    if resume_path and os.path.exists(resume_path):
        ckpt = _load(resume_path)
        policy.load_state_dict(ckpt["policy_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history   = ckpt.get("history", [])
        start_iter = ckpt.get("iteration", 0)
        print(f"Resumed from '{resume_path}' at iteration {start_iter}.")

    print(f"\n{'Iter':>5}  {'MeanRet':>9}  {'MaxRet':>8}  {'Loss':>9}"
          f"  {'Baseline':>9}  {'MeanStd':>8}  {'Steps':>7}")
    print("-" * 68)

    for it in range(start_iter, start_iter + n_iterations):
        episodes = [rollout(policy, model, data) for _ in range(episodes_per_iter)]

        ep_returns = [float(e["returns"][0]) for e in episodes]
        ep_steps   = [len(e["rewards"])      for e in episodes]
        diag       = pg_update(policy, optimizer, episodes)

        stats = {
            "iteration":   it + 1,
            "mean_return": float(np.mean(ep_returns)),
            "max_return":  float(np.max(ep_returns)),
            "mean_steps":  float(np.mean(ep_steps)),
            **diag,
        }
        history.append(stats)

        print(f"{it+1:>5}  {stats['mean_return']:>9.3f}  {stats['max_return']:>8.3f}"
              f"  {stats['loss']:>9.5f}  {stats['baseline']:>9.3f}"
              f"  {stats['mean_std']:>8.4f}  {int(stats['mean_steps']):>7}")

        if (it + 1) % checkpoint_every == 0:
            _save("pg_checkpoint.pkl", policy, optimizer, history, it + 1)
            print(f"  [checkpoint → pg_checkpoint.pkl]")

    _save(results_path, policy, optimizer, history, start_iter + n_iterations)
    print(f"\nTraining complete.  Results saved to '{results_path}'.")
    return policy, history


# ── Persistence helpers ───────────────────────────────────────────────────────

def _save(path, policy, optimizer, history, iteration):
    with open(path, "wb") as f:
        pickle.dump({
            "policy_state":    policy.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history":         history,
            "iteration":       iteration,
        }, f)


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Vanilla Policy Gradient (REINFORCE) on the MuJoCo crawler"
    )
    p.add_argument("--iters",   type=int,   default=200,   help="Gradient updates (default: 200)")
    p.add_argument("--eps",     type=int,   default=10,    help="Episodes per update (default: 10)")
    p.add_argument("--lr",      type=float, default=3e-4,  help="Adam learning rate (default: 3e-4)")
    p.add_argument("--hidden",  type=int,   default=64,    help="Hidden layer width (default: 64)")
    p.add_argument("--ckpt-every", type=int, default=20,   help="Checkpoint interval (default: 20)")
    p.add_argument("--resume",  type=str,   default=None,  help="Resume from a checkpoint .pkl")
    p.add_argument("--out",     type=str,   default="pg_results.pkl", help="Output file")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        n_iterations      = args.iters,
        episodes_per_iter = args.eps,
        lr                = args.lr,
        hidden            = args.hidden,
        checkpoint_every  = args.ckpt_every,
        results_path      = args.out,
        resume_path       = args.resume,
    )
