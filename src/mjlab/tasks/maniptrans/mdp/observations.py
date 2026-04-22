"""Observation terms for ManipTrans dexterous manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_inv, quat_mul

from .commands import ManipTransCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# --- Proprioceptive observations ---


def hand_joint_pos(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Hand finger joint positions. Shape: (B, n_dofs)."""
  hand: Entity = env.scene[asset_cfg.name]
  return hand.data.joint_pos[:, asset_cfg.joint_ids]


def hand_joint_cos_sin(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Cos and sin of finger joint positions. Shape: (B, 2*n_dofs)."""
  hand: Entity = env.scene[asset_cfg.name]
  q = hand.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.cat([torch.cos(q), torch.sin(q)], dim=-1)


def wrist_state(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Wrist state: zeros(3) + quat(4) + linvel(3) + angvel(3) = 13D.

  ManipTrans zeros wrist position to make obs env-origin-invariant.
  """
  hand: Entity = env.scene[asset_cfg.name]
  return torch.cat([
    torch.zeros_like(hand.data.root_link_pos_w),
    hand.data.root_link_quat_w,
    hand.data.root_link_lin_vel_w,
    hand.data.root_link_ang_vel_w,
  ], dim=-1)


def hand_joint_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Hand joint velocities (dq). Shape: (B, n_dofs)."""
  hand: Entity = env.scene[asset_cfg.name]
  return hand.data.joint_vel[:, asset_cfg.joint_ids]


def last_applied_action(
  env: ManagerBasedRlEnv,
  action_name: str = "maniptrans",
) -> torch.Tensor:
  """Return the first n_dofs dims of the raw policy action.

  Stage-invariant replacement for `mdp_obs.last_action`:
    - Stage 1: action_dim = n_dofs → first-n_dofs == full action (no change in size).
    - Stage 2: action_dim = 2*n_dofs → first-n_dofs == the "first half" slot of the
      residual policy's raw output (same shape as Stage 1's last_action).

  Keeps the `last_action` observation tensor at a consistent `n_dofs` width across
  stages so Stage 1 checkpoints' obs layouts slice cleanly in Stage 2 residual
  training (the `ResidualActor.get_latent` prefix slice `obs[:, :base_obs_dim]`).
  """
  action_term = env.action_manager.get_term(action_name)
  return env.action_manager.action[:, :action_term.n_dofs]


# --- MANO tracking delta observations ---


def mano_wrist_pos_delta(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Delta from robot wrist to MANO wrist target. Shape: (B, n_sides*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  delta = command.mano_wrist_pos_w - command.robot_wrist_pos_w  # (B, n_sides, 3)
  return delta.reshape(delta.shape[0], -1)


def mano_wrist_rot_delta(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Quaternion rotation delta. Shape: (B, n_sides*4)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  mano_quat = command.mano_wrist_quat_w  # (B, n_sides, 4)
  robot_quat = command.robot_wrist_quat_w  # (B, n_sides, 4)
  delta = quat_mul(mano_quat, quat_inv(robot_quat))  # (B, n_sides, 4)
  return delta.reshape(delta.shape[0], -1)


def mano_fingertip_pos_delta(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Delta from robot to MANO for all 17 tracked bodies per side.

  ManipTrans: delta_joints_pos = mano_joints - joints_state[:, 1:, :3] (17 bodies).
  We use: 12 non-tip bodies + 5 tip sites = 17 per side.
  Shape: (B, n_sides*17*3).
  """
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  parts = []
  for side in command._side_list:
    si = command._side_list.index(side)
    body_delta = command.mano_all_joints_pos_w(side) - command.robot_all_joints_pos_w(side)  # (B, 12, 3)
    tip_delta = command.mano_tip_pos_w[:, si] - command.robot_tip_pos_w[:, si]  # (B, 5, 3)
    parts.append(torch.cat([body_delta, tip_delta], dim=1))  # (B, 17, 3)
  delta = torch.stack(parts, dim=1)  # (B, n_sides, 17, 3)
  return delta.reshape(delta.shape[0], -1)


def mano_wrist_vel_delta(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Wrist velocity delta. Shape: (B, n_sides*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  delta = command.mano_wrist_vel_w - command.robot_wrist_vel_w  # (B, n_sides, 3)
  return delta.reshape(delta.shape[0], -1)


def mano_wrist_angvel_delta(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Wrist angular velocity delta. Shape: (B, n_sides*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  delta = command.mano_wrist_angvel_w - command.robot_wrist_angvel_w
  return delta.reshape(delta.shape[0], -1)


# --- Absolute MANO reference observations ---


def mano_wrist_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Absolute MANO wrist velocity. Shape: (B, n_sides*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  return command.mano_wrist_vel_w.reshape(command.mano_wrist_vel_w.shape[0], -1)


def mano_wrist_quat(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Absolute MANO wrist quaternion. Shape: (B, n_sides*4)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  return command.mano_wrist_quat_w.reshape(command.mano_wrist_quat_w.shape[0], -1)


def mano_wrist_angvel(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Absolute MANO wrist angular velocity. Shape: (B, n_sides*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  return command.mano_wrist_angvel_w.reshape(command.mano_wrist_angvel_w.shape[0], -1)


def mano_joints_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """MANO reference velocities for all 17 bodies. Shape: (B, n_sides*17*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  parts = []
  for side in command._side_list:
    si = command._side_list.index(side)
    body_vel = command.mano_all_joints_vel_w(side)  # (B, 12, 3)
    tip_vel = command.mano_tip_vel_w[:, si]  # (B, 5, 3)
    parts.append(torch.cat([body_vel, tip_vel], dim=1))  # (B, 17, 3)
  result = torch.stack(parts, dim=1)
  return result.reshape(result.shape[0], -1)


def mano_joints_vel_delta(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Delta of 17-body velocities (MANO - robot). Shape: (B, n_sides*17*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  parts = []
  for side in command._side_list:
    si = command._side_list.index(side)
    body_delta = command.mano_all_joints_vel_w(side) - command.robot_all_joints_vel_w(side)
    tip_mano_vel = command.mano_tip_vel_w[:, si]  # (B, 5, 3)
    # Tip velocity from sites — use body velocity of link2 as approximation
    tip_robot_vel = command.robot_all_joints_vel_w(side)[:, -5:]  # last 5 are link2 bodies
    # Actually tip sites don't have velocity. Use the link2 body vel for the 5 intermediates.
    # ManipTrans uses joints_state[:, 1:, 7:10] which is body lin vel for all bodies.
    # Our tips are sites, not bodies, so we approximate with link2 body vel.
    tip_delta = tip_mano_vel - tip_robot_vel
    parts.append(torch.cat([body_delta, tip_delta], dim=1))  # (B, 17, 3)
  result = torch.stack(parts, dim=1)
  return result.reshape(result.shape[0], -1)


def mano_tips_distance_obs(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Precomputed MANO tip-to-object-surface distance. Shape: (B, n_sides*5)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  return command.mano_tips_distance.reshape(command.mano_tips_distance.shape[0], -1)


# --- Object observations (Stage 2) ---


def obj_pos_relative(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Object position relative to wrist. Shape: (B, n_sides*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  delta = command.sim_obj_pos_w - command.robot_wrist_pos_w  # (B, n_sides, 3)
  return delta.reshape(delta.shape[0], -1)


def obj_quat(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Object quaternion. Shape: (B, n_sides*4)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  return command.sim_obj_quat_w.reshape(command.sim_obj_quat_w.shape[0], -1)


def obj_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Object velocity. Shape: (B, n_sides*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  return command.sim_obj_vel_w.reshape(command.sim_obj_vel_w.shape[0], -1)


def obj_angvel(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Object angular velocity. Shape: (B, n_sides*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  return command.sim_obj_angvel_w.reshape(command.sim_obj_angvel_w.shape[0], -1)


def obj_vel_delta(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Delta of object velocity (ref - sim). Shape: (B, n_sides*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  delta = command.ref_obj_vel_w - command.sim_obj_vel_w
  return delta.reshape(delta.shape[0], -1)


def obj_quat_delta(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Quaternion delta of object rotation (ref * inv(sim)). Shape: (B, n_sides*4)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  delta = quat_mul(command.ref_obj_quat_w, quat_inv(command.sim_obj_quat_w))
  return delta.reshape(delta.shape[0], -1)


def obj_angvel_delta(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Delta of object angular velocity (ref - sim). Shape: (B, n_sides*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  delta = command.ref_obj_angvel_w - command.sim_obj_angvel_w
  return delta.reshape(delta.shape[0], -1)


def obj_pos_delta(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Delta from sim object to ref object pos. Shape: (B, n_sides*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  delta = command.ref_obj_pos_w - command.sim_obj_pos_w  # (B, n_sides, 3)
  return delta.reshape(delta.shape[0], -1)


def hand_obj_distance(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Distance from object center to all 18 hand bodies per side.

  ManipTrans: obj_to_joints = norm(obj_pos - joints_state[:,:,:3]) (18 bodies).
  We use: wrist(1) + 12 non-tip bodies + 5 tip sites = 18 per side.
  """
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  parts = []
  for side in command._side_list:
    si = command._side_list.index(side)
    obj_pos = command.sim_obj_pos_w[:, si:si + 1]  # (B, 1, 3)

    wrist_pos = command.robot_wrist_pos_w[:, si:si + 1]  # (B, 1, 3)
    body_pos = command.robot_all_joints_pos_w(side)  # (B, 12, 3)
    tip_pos = command.robot_tip_pos_w[:, si]  # (B, 5, 3)

    all_pos = torch.cat([wrist_pos, body_pos, tip_pos], dim=1)  # (B, 18, 3)
    dist = torch.norm(obj_pos - all_pos, dim=-1)  # (B, 18)
    parts.append(dist)

  return torch.cat(parts, dim=-1)  # (B, n_sides * 18)


# --- Contact observations (Stage 2) ---


def _log_norm_force(force: torch.Tensor) -> torch.Tensor:
  """Apply MaskedManipulator's log-norm transform to a force tensor.

  Input:  ``force`` shape ``(..., 3)`` — raw Cartesian force vector from the
          contact sensor.
  Output: shape ``(..., 4)`` — ``[unit_direction * log(|f|+1), log(|f|+1)]``

  The transform:
    - Preserves force DIRECTION (unit vector).
    - Log-compresses MAGNITUDE so the distribution is flat enough that a
      running-stat normalizer converges to a sensible std (not near-zero).
    - Puts everything in a bounded range — `log(|f|+1)` is 0 for |f|=0, ~0.69
      for 1 N, ~2.4 for 10 N, ~4.6 for 100 N. The MLP sees values in [0, ~7]
      for typical contact magnitudes, not raw values in [0, 100] with a
      near-zero median.

  Critical for the Stage 2 residual frozen-base workflow: without this
  transform, the contact-force obs dims have a very sparse raw distribution
  (mostly 0 with rare large spikes), so rsl_rl's EmpiricalNormalization
  converges to std ≈ 0.03-0.11 on those dims. When the base is later frozen,
  a reset-spike value of 50 N maps to a z-score of ~450, the frozen MLP
  extrapolates catastrophically, and sim goes NaN within a few steps. The
  log transform flattens the distribution so the normalizer's std is ~1-2
  and even large raw spikes map to z-scores in the single digits.

  The 4th channel `log(|f|+1)` is strictly monotonic in `|f|` and > 0 iff
  |f| > 0, so binary contact checks (like `buf[..., 3] > 0` in the
  `contact_expected_but_missing` termination) still work unchanged.

  Reference: `.cc/reference/MaskedManipulator/protomotions/envs/base_env/components/humanoid_obs.py:137-140`.
  """
  norm = force.norm(dim=-1, keepdim=True)  # (..., 1)
  unit = force / (norm + 1e-6)
  log_mag = torch.log(norm + 1)
  log_xyz = unit * log_mag                 # direction * log-compressed magnitude
  return torch.cat([log_xyz, log_mag], dim=-1)  # (..., 4)


def contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Log-norm contact force from sensor: direction*log|f| + log|f| per primary.

  Shape: `(B, n_primaries * 4)`. See `_log_norm_force` for the rationale — raw
  forces get log-compressed so the normalizer's running stats stay well-behaved
  across the magnitude range.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force  # (B, n_primaries, 3)
  log_force = _log_norm_force(force)  # (B, n_primaries, 4)
  return log_force.reshape(log_force.shape[0], -1)


def contact_force_history(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  history_len: int,
) -> torch.Tensor:
  """Rolling history of per-finger log-norm contact force (4D).

  Shape: (B, history_len * n_primaries * 4).
  Buffer stored in env.extras as (B, history_len, n_primaries, 4), slot 0 is
  the oldest step and slot history_len-1 is the most recent (newer step pushed
  at the tail).

  Per-step contents: `[unit_direction * log(|f|+1), log(|f|+1)]` — same
  log-norm transform as `contact_force`, stored per history slot. The
  magnitude channel `log(|f|+1)` is > 0 iff |f| > 0, so the
  `contact_expected_but_missing` termination's `buf[..., 3] > 0` check still
  works as "any contact detected".

  Init / reset: at episode start (episode_length_buf == 1) the buffer is set
  to `[0, 0, 0, 1]` per step per finger — zero vector, log-magnitude 1.0. This
  matches the "assume recent contact" convention from ManipTrans (the rolling
  check should *not* trip `contact_missing` before the robot has had a chance
  to touch anything). The 1.0 in the log-channel is in the log-transformed
  space; it corresponds to a raw force of `exp(1) - 1 ≈ 1.72 N`, which is a
  reasonable "assume contact happened recently" magnitude.

  Consumers:
    - Policy observation: this 4D rolling history (log-scaled).
    - `contact_expected_but_missing` termination: reads the magnitude channel
      (`buf[..., 3] > 0`) to derive "was there any contact in the last N steps".
      No separate bool buffer needed.
  """
  key = f"_contact_force_history_{sensor_name}"
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force  # (B, n_primaries, 3)
  current = _log_norm_force(force)  # (B, n_primaries, 4)
  n_primaries = force.shape[1]

  if key not in env.extras:
    init = torch.zeros(
      env.num_envs, history_len, n_primaries, 4, device=env.device, dtype=torch.float
    )
    init[..., 3] = 1.0  # ManipTrans "assume recent contact" — magnitude channel
    env.extras[key] = init

  buf = env.extras[key]

  # Reset buffer for envs that just reset (episode_length_buf == 1 = first step)
  reset_mask = env.episode_length_buf == 1
  if reset_mask.any():
    buf[reset_mask] = 0.0
    buf[reset_mask, ..., 3] = 1.0

  buf = torch.cat([buf[:, 1:], current[:, None]], dim=1)  # shift + append
  env.extras[key] = buf

  return buf.reshape(env.num_envs, -1)


def ref_contact_flags(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Reference binary flag "should this finger be in contact now" per finger
  per side, from MANO preprocessing. Shape: (B, n_sides * 5).

  Raw binary (0.0 or 1.0). Matches the `contact_match` reward's flag gating
  exactly, so the policy can see the same signal the reward uses.
  """
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  flags = command.ref_contact_flags  # (B, n_sides, 5)
  return flags.reshape(flags.shape[0], -1)


# --- Future object trajectory observations (Stage 2) ---


def future_obj_pos_delta(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Delta from sim obj to next-frame ref obj pos. Shape: (B, n_sides*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  delta = command.next_obj_pos_w - command.sim_obj_pos_w
  return delta.reshape(delta.shape[0], -1)


def future_obj_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Next-frame ref obj velocity. Shape: (B, n_sides*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  return command.next_obj_vel_w.reshape(command.next_obj_vel_w.shape[0], -1)
