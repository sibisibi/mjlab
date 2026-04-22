"""Periodic Stage 2 evaluator — rolls out the residual policy on a free-object
eval env and returns success rate + per-metric time-averaged errors.

Success definition (from `eval/score.py:199–237`, the paper-matching offline
scorer — ported verbatim so in-training eval matches post-training eval):
  For each env, time-average `obj_pos_err` / `tip_err` / `obj_rot_err` over the
  window `[grace_steps, T_m − tail_skip_steps)`. Reduce across sides with max
  (bimanual: worst side carries). Success at threshold k requires ALL of:
    obj_pos_avg  <  k · 3 cm
    tip_avg      <  k · 6 cm
    obj_rot_avg  <  k · 30°
  SR@k = fraction of envs meeting the criterion.

  Default k set logged every eval: {0.5, 1.0, 1.5, 2.0, 3.0}.
  Headline: k=1.0 (paper-strict ManipTrans thresholds).

Also returns continuous E_* metrics (wrist_pos_cm, wrist_rot_deg, fingertip_cm,
obj_pos_cm, obj_rot_deg) as mean-over-envs of per-env max-over-sides
time-averaged error.

Mirrors `stage1_scorer.py` in structure (evaluate_* + log_*_to_wandb split).
"""

from __future__ import annotations

import math
from typing import Any, Callable, cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.maniptrans.mdp.commands import ManipTransCommand

# Paper-strict thresholds at k=1.0 (ManipTrans / 260416 reproduction).
_TH_OBJ_POS_M = 0.03     # 3 cm
_TH_TIP_POS_M = 0.06     # 6 cm
_TH_OBJ_ROT_DEG = 30.0   # 30 degrees


def _rotmat_axis_angle_deg(R_diff: torch.Tensor) -> torch.Tensor:
  trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
  cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
  return torch.acos(cos_theta) * (180.0 / math.pi)


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return torch.stack([
    torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
    torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1),
    torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1),
  ], dim=-2)


def _k_to_key(k: float) -> str:
  return f"SR_at_{str(k).replace('.', 'p')}"


def evaluate_stage2(
  env: ManagerBasedRlEnv,
  env_wrapped: RslRlVecEnvWrapper,
  policy: Callable[[Any], torch.Tensor],
  device: str,
  ks: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0),
  grace_steps: int = 15,
  tail_skip_steps: int = 15,
) -> dict[str, Any]:
  """Run one pass of eval rollouts on the free-object Stage 2 env.

  Expects env built with:
    - `pin_objects=False` (objects are free, policy must cause the motion)
    - `sampling_mode="start"` (every env starts from frame 0)
    - all terminations cleared (rollouts always reach motion end)
    - observation corruption off
    - `num_envs` = desired eval rollout count (e.g. 100)

  Returns:
    {
      "global": {
        # SR at each k, as float in [0, 1]
        "SR_at_0p5", "SR_at_1p0", "SR_at_1p5", "SR_at_2p0", "SR_at_3p0",
        # Continuous metrics (mean over envs of per-env max-over-sides
        # time-averaged error)
        "E_wrist_pos_cm", "E_wrist_rot_deg", "E_fingertip_cm",
        "E_obj_pos_cm", "E_obj_rot_deg",
        # Bookkeeping
        "n_envs", "n_valid_envs", "n_sides",
        "motion_T", "eval_window_len",
        "n_k1_passed",
      }
    }

  Note: this evaluator can in principle handle multi-trajectory eval envs
  (via the per-env `T_m_per_env` and `ended` tracking), but the Stage 2
  per-traj sweep always uses 1 trajectory per run.
  """
  cmd = cast(ManipTransCommand, env.command_manager.get_term("motion"))
  motion = cmd.motion
  N = env.num_envs
  n_sides = len(cmd._side_list)
  T_max = int(motion.time_step_totals.max().item())

  obs, _ = env_wrapped.reset()
  initial_traj_idx = cmd.env_traj_idx.clone()             # (N,)
  T_m_per_env = motion.time_step_totals[initial_traj_idx] # (N,)

  # Per-env, per-side accumulators — shape (N, n_sides).
  sum_wrist_pos = torch.zeros(N, n_sides, device=device)
  sum_wrist_rot = torch.zeros(N, n_sides, device=device)
  sum_tip_pos   = torch.zeros(N, n_sides, device=device)
  sum_obj_pos   = torch.zeros(N, n_sides, device=device)
  sum_obj_rot   = torch.zeros(N, n_sides, device=device)
  step_count    = torch.zeros(N, dtype=torch.long, device=device)
  ended         = torch.zeros(N, dtype=torch.bool, device=device)

  with torch.no_grad():
    for _step in range(T_max):
      prev_time_steps = cmd.time_steps.clone()
      actions = policy(obs)
      obs, _rewards, _dones, _extras = env_wrapped.step(actions)

      # Detect traj wrap — after wrap, time_steps resets to 0. Envs that
      # wrapped stop contributing to this eval from here on.
      wrapped_now = cmd.time_steps < prev_time_steps
      ended = ended | wrapped_now

      # Per-env, per-side metric tensors, all (N, n_sides).
      wrist_pos_err = torch.norm(
        cmd.robot_wrist_pos_w - cmd.mano_wrist_pos_w, dim=-1
      )
      R_diff_wrist = cmd.mano_wrist_rot_w @ cmd.robot_wrist_rot_w.transpose(-1, -2)
      wrist_rot_err_deg = _rotmat_axis_angle_deg(R_diff_wrist)
      # tip err: ||mano_tip_pos - robot_tip_pos||, then mean over the 5 fingers
      tip_err_mean = torch.norm(
        cmd.mano_tip_pos_w - cmd.robot_tip_pos_w, dim=-1
      ).mean(dim=-1)                                       # (N, n_sides)
      obj_pos_err = torch.norm(
        cmd.ref_obj_pos_w - cmd.sim_obj_pos_w, dim=-1
      )
      sim_R = _quat_to_rotmat(cmd.sim_obj_quat_w)
      R_diff_obj = cmd.ref_obj_rotmat_w @ sim_R.transpose(-1, -2)
      obj_rot_err_deg = _rotmat_axis_angle_deg(R_diff_obj)

      in_window = (
        (cmd.time_steps >= grace_steps)
        & (cmd.time_steps < (T_m_per_env - tail_skip_steps))
      )
      valid = in_window & ~ended                            # (N,)
      valid_side = valid.unsqueeze(-1)                      # (N, 1)

      zeros = torch.zeros_like(sum_wrist_pos)
      sum_wrist_pos = sum_wrist_pos + torch.where(valid_side, wrist_pos_err, zeros)
      sum_wrist_rot = sum_wrist_rot + torch.where(valid_side, wrist_rot_err_deg, zeros)
      sum_tip_pos   = sum_tip_pos   + torch.where(valid_side, tip_err_mean, zeros)
      sum_obj_pos   = sum_obj_pos   + torch.where(valid_side, obj_pos_err, zeros)
      sum_obj_rot   = sum_obj_rot   + torch.where(valid_side, obj_rot_err_deg, zeros)
      step_count = step_count + valid.long()

      # Early exit once all envs finished their initial traj's eval window.
      if bool(ended.all().item()):
        break

  valid_env_mask = step_count > 0
  n_valid = int(valid_env_mask.sum().item())
  count_safe = step_count.clamp_min(1).float()              # (N,)

  # Time-average per (env, side).
  avg_wrist_pos = sum_wrist_pos / count_safe.unsqueeze(-1)  # m
  avg_wrist_rot = sum_wrist_rot / count_safe.unsqueeze(-1)  # deg
  avg_tip_pos   = sum_tip_pos   / count_safe.unsqueeze(-1)  # m
  avg_obj_pos   = sum_obj_pos   / count_safe.unsqueeze(-1)  # m
  avg_obj_rot   = sum_obj_rot   / count_safe.unsqueeze(-1)  # deg

  # Reduce across sides: max (bimanual worst-side carries).
  per_env_wrist_pos = avg_wrist_pos.max(dim=-1).values      # (N,) m
  per_env_wrist_rot = avg_wrist_rot.max(dim=-1).values      # deg
  per_env_tip_pos   = avg_tip_pos.max(dim=-1).values        # m
  per_env_obj_pos   = avg_obj_pos.max(dim=-1).values        # m
  per_env_obj_rot   = avg_obj_rot.max(dim=-1).values        # deg

  # Continuous E_* headline: mean over valid envs, converted to cm/deg.
  if n_valid > 0:
    E_wrist_pos_cm  = float(per_env_wrist_pos[valid_env_mask].mean().item()) * 100.0
    E_wrist_rot_deg = float(per_env_wrist_rot[valid_env_mask].mean().item())
    E_fingertip_cm  = float(per_env_tip_pos[valid_env_mask].mean().item())   * 100.0
    E_obj_pos_cm    = float(per_env_obj_pos[valid_env_mask].mean().item())   * 100.0
    E_obj_rot_deg   = float(per_env_obj_rot[valid_env_mask].mean().item())
  else:
    E_wrist_pos_cm = E_wrist_rot_deg = E_fingertip_cm = float("nan")
    E_obj_pos_cm = E_obj_rot_deg = float("nan")

  # SR@k — failure is ANY of obj_pos / tip / obj_rot exceeding the threshold.
  n_k = len(ks)
  sr_values: list[float] = [0.0] * n_k
  n_k1_passed = 0

  if n_valid > 0:
    k_tensor = torch.tensor(list(ks), device=device)
    obp = per_env_obj_pos[valid_env_mask].unsqueeze(-1)     # (n_valid, 1)
    tp  = per_env_tip_pos[valid_env_mask].unsqueeze(-1)
    obr = per_env_obj_rot[valid_env_mask].unsqueeze(-1)
    fail_obj_pos = obp > (k_tensor * _TH_OBJ_POS_M)
    fail_tip_pos = tp  > (k_tensor * _TH_TIP_POS_M)
    fail_obj_rot = obr > (k_tensor * _TH_OBJ_ROT_DEG)
    fail_mask = fail_obj_pos | fail_tip_pos | fail_obj_rot  # (n_valid, n_k)
    success_rates = (~fail_mask).float().mean(dim=0)        # (n_k,)
    sr_values = [float(x.item()) for x in success_rates]

    if 1.0 in ks:
      k1_idx = list(ks).index(1.0)
      n_k1_passed = int((~fail_mask[:, k1_idx]).sum().item())

  sr_dict: dict[str, float] = {
    _k_to_key(k): v for k, v in zip(ks, sr_values)
  }

  motion_T = int(T_m_per_env[0].item()) if N > 0 else 0
  window_len = max(0, motion_T - grace_steps - tail_skip_steps)

  return {
    "global": {
      **sr_dict,
      "E_wrist_pos_cm":  E_wrist_pos_cm,
      "E_wrist_rot_deg": E_wrist_rot_deg,
      "E_fingertip_cm":  E_fingertip_cm,
      "E_obj_pos_cm":    E_obj_pos_cm,
      "E_obj_rot_deg":   E_obj_rot_deg,
      "n_envs":          N,
      "n_valid_envs":    n_valid,
      "n_sides":         n_sides,
      "motion_T":        motion_T,
      "eval_window_len": window_len,
      "n_k1_passed":     n_k1_passed,
    }
  }


def log_stage2_eval_to_wandb(
  metrics: dict[str, Any],
  iter_idx: int,
  wandb_module=None,
  k_headline: float = 1.0,
) -> None:
  """Flatten + log Stage 2 eval metrics to wandb. Call after evaluate_stage2()."""
  import wandb as _wandb
  w = wandb_module if wandb_module is not None else _wandb
  if w.run is None:
    return

  g = metrics["global"]
  headline_key = _k_to_key(k_headline)

  flat: dict[str, float] = {
    "eval/success_rate":     g.get(headline_key, float("nan")),
    "eval/E_obj_pos_cm":     g["E_obj_pos_cm"],
    "eval/E_obj_rot_deg":    g["E_obj_rot_deg"],
    "eval/E_fingertip_cm":   g["E_fingertip_cm"],
    "eval/E_wrist_pos_cm":   g["E_wrist_pos_cm"],
    "eval/E_wrist_rot_deg":  g["E_wrist_rot_deg"],
    "eval/n_envs":           g["n_envs"],
    "eval/n_valid_envs":     g["n_valid_envs"],
    "eval/motion_T":         g["motion_T"],
    "eval/eval_window_len":  g["eval_window_len"],
    "eval/n_k1_passed":      g["n_k1_passed"],
  }
  for key, val in g.items():
    if key.startswith("SR_at_"):
      flat[f"eval/{key}"] = val

  w.log(flat, step=iter_idx)
