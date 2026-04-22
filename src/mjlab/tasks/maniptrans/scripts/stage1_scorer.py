"""Periodic Stage 1 evaluator — runs rollouts on an eval env and returns
per-trajectory hand-tracking metrics.

Decision metric: E_fingertip (cm). Also logs E_wrist_pos (cm), E_wrist_rot (deg).
For each rollout, errors are time-averaged over the window
`[grace_steps, T_m - tail_skip_steps)` — exactly once per env, over that env's
*initial* trajectory (env-internal wraps after T_m are ignored).

Aggregation per trajectory: mean / median / min / max over the 20-ish rollouts
that landed on that trajectory. Global scalar: mean-of-per-traj-means
(equal-weighted across trajectories — robust to a few hard trajs dominating).
"""

from __future__ import annotations

import math
from typing import Any, Callable, cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.maniptrans.mdp.commands import ManipTransCommand


def _rotmat_axis_angle_deg(R_diff: torch.Tensor) -> torch.Tensor:
  trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
  cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
  return torch.acos(cos_theta) * (180.0 / math.pi)


def evaluate_stage1(
  env: ManagerBasedRlEnv,
  env_wrapped: RslRlVecEnvWrapper,
  policy: Callable[[Any], torch.Tensor],
  device: str,
  grace_steps: int = 15,
  tail_skip_steps: int = 15,
) -> dict[str, Any]:
  """Run one pass of eval rollouts and return per-traj / global metrics.

  Expects the env to be built with `sampling_mode="start"` and with terminations
  that don't truncate imitation-quality rollouts (fingertip_diverged,
  dof_vel_sanity, contact_missing should be off). Each env plays its
  randomly-assigned trajectory from frame 0; errors are accumulated only on
  the initial trajectory and within the eval window.

  Returns a nested dict:
    {
      "global": {"wrist_pos_cm", "wrist_rot_deg", "tip_pos_cm",
                 "n_trajectories_evaluated"},
      "per_traj": {
        <traj_idx>: {
          "n_rollouts", "wrist_pos_cm": {mean/median/min/max}, ...
        }
      }
    }
  """
  cmd = cast(ManipTransCommand, env.command_manager.get_term("motion"))
  motion = cmd.motion
  M = motion.num_trajectories
  N = env.num_envs
  T_max = int(motion.time_step_totals.max().item())
  n_sides = len(cmd._side_list)

  # Reset — each env gets a random traj via _resample_command (or M=1 trivially).
  obs, _ = env_wrapped.reset()
  initial_traj_idx = cmd.env_traj_idx.clone()  # (N,)
  T_m_per_env = motion.time_step_totals[initial_traj_idx]  # (N,)

  # Per-env accumulators (error magnitudes are side-averaged before storing).
  sum_wrist_pos = torch.zeros(N, device=device)
  sum_wrist_rot = torch.zeros(N, device=device)
  sum_tip_pos = torch.zeros(N, device=device)
  step_count = torch.zeros(N, dtype=torch.long, device=device)
  ended = torch.zeros(N, dtype=torch.bool, device=device)

  with torch.no_grad():
    for _step in range(T_max):
      prev_time_steps = cmd.time_steps.clone()
      actions = policy(obs)
      obs, _rewards, _dones, _extras = env_wrapped.step(actions)

      # Detect traj wrap: after wrap, time_steps resets to 0 while previous was ≈T_m.
      # A wrapped env should stop contributing to this eval from here on.
      wrapped_now = cmd.time_steps < prev_time_steps
      ended = ended | wrapped_now

      # Errors on current frame (all in world frame; env.scene.env_origins already
      # baked into mano_* properties). mean over sides so the reported number is
      # a single scalar per env regardless of single/bimanual config.
      wrist_pos_err = torch.norm(
        cmd.robot_wrist_pos_w - cmd.mano_wrist_pos_w, dim=-1
      ).mean(dim=-1)  # (N,) meters
      R_diff = cmd.mano_wrist_rot_w @ cmd.robot_wrist_rot_w.transpose(-1, -2)
      wrist_rot_err_deg = _rotmat_axis_angle_deg(R_diff).mean(dim=-1)  # (N,) deg
      tip_err = torch.norm(
        cmd.mano_tip_pos_w - cmd.robot_tip_pos_w, dim=-1
      ).mean(dim=(-1, -2))  # (N,) meters, mean over (sides, 5 tips)

      in_window = (
        (cmd.time_steps >= grace_steps)
        & (cmd.time_steps < (T_m_per_env - tail_skip_steps))
      )
      valid = in_window & ~ended

      zeros = torch.zeros_like(wrist_pos_err)
      sum_wrist_pos = sum_wrist_pos + torch.where(valid, wrist_pos_err, zeros)
      sum_wrist_rot = sum_wrist_rot + torch.where(valid, wrist_rot_err_deg, zeros)
      sum_tip_pos = sum_tip_pos + torch.where(valid, tip_err, zeros)
      step_count = step_count + valid.long()

      # Early exit: all envs finished their initial traj's eval window.
      if bool(ended.all().item()):
        break

  # Per-env time averages. Envs with zero valid steps (e.g. eval window smaller
  # than their traj length minus grace+tail — shouldn't happen with T_m>30 but
  # guard against div-by-zero) are excluded via the step_count > 0 mask below.
  count_safe = step_count.clamp_min(1).float()
  per_env_wrist_pos_cm = (sum_wrist_pos / count_safe) * 100.0  # (N,)
  per_env_wrist_rot_deg = sum_wrist_rot / count_safe
  per_env_tip_pos_cm = (sum_tip_pos / count_safe) * 100.0

  valid_env_mask = step_count > 0

  def _summary(x: torch.Tensor) -> dict[str, float]:
    if x.numel() == 0:
      return {"mean": float("nan"), "median": float("nan"),
              "min": float("nan"), "max": float("nan")}
    return {
      "mean": float(x.mean().item()),
      "median": float(x.median().item()),
      "min": float(x.min().item()),
      "max": float(x.max().item()),
    }

  per_traj: dict[int, dict[str, Any]] = {}
  for m in range(M):
    env_mask = (initial_traj_idx == m) & valid_env_mask
    n_rollouts = int(env_mask.sum().item())
    if n_rollouts == 0:
      continue
    per_traj[m] = {
      "n_rollouts": n_rollouts,
      "wrist_pos_cm": _summary(per_env_wrist_pos_cm[env_mask]),
      "wrist_rot_deg": _summary(per_env_wrist_rot_deg[env_mask]),
      "tip_pos_cm": _summary(per_env_tip_pos_cm[env_mask]),
    }

  # Global: mean of per-traj means (equal-weighted across trajectories).
  if per_traj:
    g_wrist_pos = sum(pt["wrist_pos_cm"]["mean"] for pt in per_traj.values()) / len(per_traj)
    g_wrist_rot = sum(pt["wrist_rot_deg"]["mean"] for pt in per_traj.values()) / len(per_traj)
    g_tip_pos = sum(pt["tip_pos_cm"]["mean"] for pt in per_traj.values()) / len(per_traj)
  else:
    g_wrist_pos = g_wrist_rot = g_tip_pos = float("nan")

  return {
    "global": {
      "wrist_pos_cm": g_wrist_pos,
      "wrist_rot_deg": g_wrist_rot,
      "tip_pos_cm": g_tip_pos,
      "n_trajectories_evaluated": len(per_traj),
      "n_envs": N,
      "n_sides": n_sides,
    },
    "per_traj": per_traj,
  }


def log_eval_to_wandb(
  metrics: dict[str, Any],
  iter_idx: int,
  wandb_module=None,
) -> None:
  """Flatten + log metrics to wandb. Call after evaluate_stage1()."""
  import wandb as _wandb
  w = wandb_module if wandb_module is not None else _wandb
  if w.run is None:
    return

  g = metrics["global"]
  flat: dict[str, float] = {
    "eval/E_fingertip_cm": g["tip_pos_cm"],  # decision metric
    "eval/E_wrist_pos_cm": g["wrist_pos_cm"],
    "eval/E_wrist_rot_deg": g["wrist_rot_deg"],
    "eval/n_trajectories_evaluated": g["n_trajectories_evaluated"],
  }
  # Per-traj breakdown under eval/traj_<idx>/...
  for m, pt in metrics["per_traj"].items():
    flat[f"eval/traj_{m}/tip_pos_cm_mean"] = pt["tip_pos_cm"]["mean"]
    flat[f"eval/traj_{m}/tip_pos_cm_max"] = pt["tip_pos_cm"]["max"]
    flat[f"eval/traj_{m}/wrist_pos_cm_mean"] = pt["wrist_pos_cm"]["mean"]
    flat[f"eval/traj_{m}/wrist_rot_deg_mean"] = pt["wrist_rot_deg"]["mean"]
    flat[f"eval/traj_{m}/n_rollouts"] = pt["n_rollouts"]

  w.log(flat, step=iter_idx)
