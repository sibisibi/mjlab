"""Observation terms for ManipTrans dexterous manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_inv, quat_mul

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


def _to_wrist_frame(
  vec_w: torch.Tensor, command: ManipTransCommand
) -> torch.Tensor:
  """Rotate world-frame 3-vectors into per-side wrist frame.

  vec_w shape: (B, n_sides, 3) or (B, n_sides, K, 3) for any K. Returns same
  shape with each side's vectors expressed in that side's wrist frame.

  Wrist frame matters because the policy plans against its own kinematic
  reference, not the world. For the same object pose, a wrist rotated 90 deg
  yaw should see the same wrist-frame inputs — not a rotated world-frame copy.
  """
  wrist_quat = command.robot_wrist_quat_w  # (B, n_sides, 4)
  if vec_w.dim() == wrist_quat.dim():
    return quat_apply_inverse(wrist_quat, vec_w)
  extra = vec_w.dim() - wrist_quat.dim()
  for _ in range(extra):
    wrist_quat = wrist_quat.unsqueeze(2)
  wrist_quat = wrist_quat.expand(*vec_w.shape[:-1], 4)
  return quat_apply_inverse(wrist_quat, vec_w)


def _quat_to_wrist_frame(
  quat_w: torch.Tensor, command: ManipTransCommand
) -> torch.Tensor:
  """Express a per-side world-frame quaternion in the per-side wrist frame.

  q_wrist_obj = quat_inv(q_wrist_world) @ q_obj_world. Shape: (B, n_sides, 4).
  """
  return quat_mul(quat_inv(command.robot_wrist_quat_w), quat_w)


def mano_joints_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """MANO reference velocities for all 17 bodies, in wrist frame.

  Shape: (B, n_sides*17*3).
  """
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  parts = []
  for side in command._side_list:
    si = command._side_list.index(side)
    body_vel = command.mano_all_joints_vel_w(side)  # (B, 12, 3)
    tip_vel = command.mano_tip_vel_w[:, si]  # (B, 5, 3)
    parts.append(torch.cat([body_vel, tip_vel], dim=1))  # (B, 17, 3)
  result = torch.stack(parts, dim=1)  # (B, n_sides, 17, 3)
  result = _to_wrist_frame(result, command)
  return result.reshape(result.shape[0], -1)


def mano_joints_vel_delta(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Delta of 17-body velocities (MANO - robot), in wrist frame.

  Shape: (B, n_sides*17*3).
  """
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
  result = torch.stack(parts, dim=1)  # (B, n_sides, 17, 3)
  result = _to_wrist_frame(result, command)
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
  """Object position relative to wrist, in wrist frame. Shape: (B, n_sides*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  delta = command.sim_obj_pos_w - command.robot_wrist_pos_w  # (B, n_sides, 3)
  delta = _to_wrist_frame(delta, command)
  return delta.reshape(delta.shape[0], -1)


def obj_quat(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Object quaternion expressed in wrist frame. Shape: (B, n_sides*4).

  q_wrist_obj = quat_inv(q_wrist_world) @ q_obj_world. The actor sees the
  object's orientation as it would appear from the palm frame, so a wrist yaw
  does not change the encoding for a fixed object.
  """
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  q = _quat_to_wrist_frame(command.sim_obj_quat_w, command)
  return q.reshape(q.shape[0], -1)


def obj_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Object linear velocity in wrist frame. Shape: (B, n_sides*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  v = _to_wrist_frame(command.sim_obj_vel_w, command)
  return v.reshape(v.shape[0], -1)


def obj_angvel(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Object angular velocity in wrist frame. Shape: (B, n_sides*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  w = _to_wrist_frame(command.sim_obj_angvel_w, command)
  return w.reshape(w.shape[0], -1)


def obj_vel_delta(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Delta of object linear velocity (ref - sim) in wrist frame.

  Shape: (B, n_sides*3).
  """
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  delta = command.ref_obj_vel_w - command.sim_obj_vel_w
  delta = _to_wrist_frame(delta, command)
  return delta.reshape(delta.shape[0], -1)


def obj_quat_delta(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Quaternion delta from sim to ref object orientation, in wrist frame.

  Computed as q_ref_in_wrist @ quat_inv(q_sim_in_wrist), where each side's
  obj quat is first re-expressed in that side's wrist frame. The axis of the
  returned rotation lives in wrist coordinates so wrist rotations don't
  reshape the error signal.

  Shape: (B, n_sides*4).
  """
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  wrist_inv = quat_inv(command.robot_wrist_quat_w)
  sim_in_wrist = quat_mul(wrist_inv, command.sim_obj_quat_w)
  ref_in_wrist = quat_mul(wrist_inv, command.ref_obj_quat_w)
  delta = quat_mul(ref_in_wrist, quat_inv(sim_in_wrist))
  return delta.reshape(delta.shape[0], -1)


def obj_angvel_delta(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Delta of object angular velocity (ref - sim) in wrist frame.

  Shape: (B, n_sides*3).
  """
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  delta = command.ref_obj_angvel_w - command.sim_obj_angvel_w
  delta = _to_wrist_frame(delta, command)
  return delta.reshape(delta.shape[0], -1)


def obj_pos_delta(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Delta from sim to ref object pos, in wrist frame. Shape: (B, n_sides*3)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  delta = command.ref_obj_pos_w - command.sim_obj_pos_w  # (B, n_sides, 3)
  delta = _to_wrist_frame(delta, command)
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


def _force_to_wrist_frame(
  force: torch.Tensor, command: ManipTransCommand, side: str
) -> torch.Tensor:
  """Rotate a per-finger world-frame force tensor into the side's wrist frame.

  force shape: (B, n_primaries, 3). Returns (B, n_primaries, 3).
  """
  si = command._side_list.index(side)
  wrist_quat = command.robot_wrist_quat_w[:, si:si + 1]  # (B, 1, 4)
  wrist_quat = wrist_quat.expand(force.shape[0], force.shape[1], 4)
  return quat_apply_inverse(wrist_quat, force)


def contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  side: str,
) -> torch.Tensor:
  """Log-norm contact force from sensor in wrist frame.

  Per-finger 4D `[unit_dir * log(|f|+1), log(|f|+1)]` — direction expressed in
  the per-side wrist frame so the actor reads the same encoding for the same
  contact regardless of wrist orientation. Shape: ``(B, n_primaries * 4)``.

  See `_log_norm_force` for the magnitude-compression rationale.
  """
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force  # (B, n_primaries, 3)
  force = _force_to_wrist_frame(force, command, side)
  log_force = _log_norm_force(force)  # (B, n_primaries, 4)
  return log_force.reshape(log_force.shape[0], -1)


def contact_force_history(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  side: str,
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
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force  # (B, n_primaries, 3)
  force = _force_to_wrist_frame(force, command, side)
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


def contact_found_history(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  history_len: int,
) -> torch.Tensor:
  """Rolling history of per-finger `found` flag from the penetration sensor.

  Shape: (B, history_len * n_primaries) — float buffer with {0.0, 1.0} entries.
  Stored in env.extras[f"_contact_found_history_{sensor_name}"] as
  (B, history_len, n_primaries). Slot 0 is oldest, slot history_len-1 is the
  most recent step.

  `found` comes from the penetration (mindist) sensor — same signal the
  `contact_point_match_reward` already uses. A value of 1.0 means the
  narrowphase produced a contact between the fingertip site and the object
  body on that step. Independent of contact force magnitude, so a policy
  gripping firmly vs. gently both register the same.

  Init / reset: at episode start the buffer is filled with 1.0 ("assume
  recent contact") so the `contact_expected_but_missing` termination does
  not trip before the robot has had a chance to make contact.
  """
  key = f"_contact_found_history_{sensor_name}"
  sensor: ContactSensor = env.scene[sensor_name]
  current = (sensor.data.found > 0).to(torch.float)  # (B, n_primaries)
  n_primaries = current.shape[1]

  if key not in env.extras:
    env.extras[key] = torch.ones(
      env.num_envs, history_len, n_primaries, device=env.device, dtype=torch.float
    )

  buf = env.extras[key]
  reset_mask = env.episode_length_buf == 1
  if reset_mask.any():
    buf[reset_mask] = 1.0

  buf = torch.cat([buf[:, 1:], current[:, None]], dim=1)
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


def tip_penetration(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  dist_clamp_min: float = -0.02,
  dist_clamp_max: float = 0.05,
) -> torch.Tensor:
  """Per-finger signed penetration distance + narrowphase-found flag.

  Shape: (B, n_primaries * 2) where n_primaries=5 fingertips and the trailing
  pair is `[dist_clamped, found]`:
    - dist: signed distance from the penetration (mindist) sensor (< 0 when
      overlap), clamped to [dist_clamp_min, dist_clamp_max] to bound the
      EmpiricalNormalization stats against transient resets.
    - found: float in {0.0, 1.0} — narrowphase produced contact between the
      fingertip and the object body this step.

  Why expose these to the actor: contact force is zero in the regime that
  matters most for approach (the last 1-5 cm before contact). `dist` stays
  continuous through that regime; `found` is the same binary signal the
  contact_point_match_reward gates on. Together they make the pre-contact
  approach phase observable instead of inferred.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  dist = sensor.data.dist.clamp(dist_clamp_min, dist_clamp_max)  # (B, n_primaries)
  found = (sensor.data.found > 0).to(dist.dtype)  # (B, n_primaries)
  out = torch.stack([dist, found], dim=-1)  # (B, n_primaries, 2)
  return out.reshape(out.shape[0], -1)


# --- Future object trajectory observations (Stage 2) ---


def future_obj_pos_delta(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Delta from sim obj to next-frame ref obj pos, in wrist frame.

  Shape: (B, n_sides*3).
  """
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  delta = command.next_obj_pos_w - command.sim_obj_pos_w
  delta = _to_wrist_frame(delta, command)
  return delta.reshape(delta.shape[0], -1)


def future_obj_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Next-frame ref obj linear velocity, in wrist frame.

  Shape: (B, n_sides*3).
  """
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  v = _to_wrist_frame(command.next_obj_vel_w, command)
  return v.reshape(v.shape[0], -1)
