"""Reward terms for ManipTrans dexterous manipulation.

All tracking rewards are per-side (take `side` parameter).
For bimanual, config has separate r_* and l_* entries.
Rewards are summed across sides (ManipTrans convention), not averaged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import FINGER_NAMES, ManipTransCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _cmd(env: ManagerBasedRlEnv, command_name: str) -> ManipTransCommand:
  return cast(ManipTransCommand, env.command_manager.get_term(command_name))


def _side_idx(command: ManipTransCommand, side: str) -> int:
  return command._side_list.index(side)


# --- Per-side MANO tracking rewards ---


def mano_fingertip_pos_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  side: str,
  finger: str,
  scale: float,
) -> torch.Tensor:
  """exp(-scale * ||mano_tip - robot_tip||) for one finger on one side."""
  command = _cmd(env, command_name)
  si = _side_idx(command, side)
  fi = FINGER_NAMES.index(finger)
  error = torch.norm(
    command.mano_tip_pos_w[:, si, fi] - command.robot_tip_pos_w[:, si, fi], dim=-1
  )
  return torch.exp(-scale * error)


def mano_wrist_vel_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  side: str,
  scale: float,
) -> torch.Tensor:
  """exp(-scale * mean(|mano_vel - robot_vel|)) for one side."""
  command = _cmd(env, command_name)
  si = _side_idx(command, side)
  error = torch.mean(
    torch.abs(command.mano_wrist_vel_w[:, si] - command.robot_wrist_vel_w[:, si]), dim=-1
  )
  return torch.exp(-scale * error)


def mano_wrist_angvel_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  side: str,
  scale: float,
) -> torch.Tensor:
  """exp(-scale * mean(|mano_angvel - robot_angvel|)) for one side."""
  command = _cmd(env, command_name)
  si = _side_idx(command, side)
  error = torch.mean(
    torch.abs(command.mano_wrist_angvel_w[:, si] - command.robot_wrist_angvel_w[:, si]), dim=-1
  )
  return torch.exp(-scale * error)


# --- Per-side wrist tracking rewards ---


def mano_wrist_pos_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  side: str,
  scale: float,
) -> torch.Tensor:
  """exp(-scale * ||mano_wrist_pos - robot_wrist_pos||) for one side."""
  command = _cmd(env, command_name)
  si = _side_idx(command, side)
  error = torch.norm(
    command.mano_wrist_pos_w[:, si] - command.robot_wrist_pos_w[:, si], dim=-1
  )
  return torch.exp(-scale * error)


def mano_wrist_rot_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  side: str,
  scale: float,
) -> torch.Tensor:
  """exp(-scale * quat_angle(mano_wrist, robot_wrist)) for one side."""
  command = _cmd(env, command_name)
  si = _side_idx(command, side)
  error = quat_error_magnitude(
    command.mano_wrist_quat_w[:, si], command.robot_wrist_quat_w[:, si]
  )
  return torch.exp(-scale * error)


def power_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  action_name: str,
  side: str,
  scale: float,
) -> torch.Tensor:
  """exp(-scale * sum(|torque * velocity|)) for finger joints, one side."""
  command = _cmd(env, command_name)
  act_term = env.action_manager._terms[action_name]
  finger_ids = act_term._finger_ids
  # Split finger IDs by side (first half = right, second half = left)
  n_per_side = len(finger_ids) // len(command._side_list)
  si = _side_idx(command, side)
  side_finger_ids = finger_ids[si * n_per_side:(si + 1) * n_per_side]
  torque = command.robot.data.qfrc_actuator[:, side_finger_ids]
  vel = command.robot.data.joint_vel[:, side_finger_ids]
  power = torch.sum(torch.abs(torque * vel), dim=-1)
  return torch.exp(-scale * power)


def wrist_power_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  action_name: str,
  side: str,
  scale: float,
) -> torch.Tensor:
  """exp(-scale * sum(|torque * velocity|)) for wrist joints, one side."""
  command = _cmd(env, command_name)
  act_term = env.action_manager._terms[action_name]
  wrist_ids = act_term._wrist_ids
  n_per_side = len(wrist_ids) // len(command._side_list)
  si = _side_idx(command, side)
  side_wrist_ids = wrist_ids[si * n_per_side:(si + 1) * n_per_side]
  torque = command.robot.data.qfrc_actuator[:, side_wrist_ids]
  vel = command.robot.data.joint_vel[:, side_wrist_ids]
  power = torch.sum(torch.abs(torque * vel), dim=-1)
  return torch.exp(-scale * power)


# --- Per-side joint level tracking rewards ---


def mano_level_pos_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  side: str,
  level: int,
  scale: float,
) -> torch.Tensor:
  """exp(-scale * mean(||mano_joint - robot_body||)) for level 1 or 2, one side."""
  command = _cmd(env, command_name)
  mano_pos = command.mano_level_pos_w(side, level)  # (B, 5, 3)
  robot_pos = command.robot_level_pos_w(side, level)  # (B, 5, 3)
  error = torch.norm(mano_pos - robot_pos, dim=-1)  # (B, 5)
  return torch.exp(-scale * error.mean(dim=-1))




def joints_vel_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  side: str,
  scale: float,
) -> torch.Tensor:
  """exp(-scale * mean(|mano_vel - robot_vel|)) for all 17 bodies, one side.

  ManipTrans: reward_joints_vel = exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(-1))
  """
  command = _cmd(env, command_name)
  si = _side_idx(command, side)
  body_delta = command.mano_all_joints_vel_w(side) - command.robot_all_joints_vel_w(side)  # (B, 12, 3)
  tip_mano_vel = command.mano_tip_vel_w[:, si]  # (B, 5, 3)
  tip_robot_vel = command.robot_all_joints_vel_w(side)[:, -5:]  # link2 body vel as approx
  tip_delta = tip_mano_vel - tip_robot_vel  # (B, 5, 3)
  all_delta = torch.cat([body_delta, tip_delta], dim=1)  # (B, 17, 3)
  return torch.exp(-scale * all_delta.abs().mean(dim=-1).mean(dim=-1))


# --- Per-side contact rewards (Stage 2) ---


def contact_point_match_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  side: str,
  finger: str,
  beta: float,
  A: float,
  eps: float,
) -> torch.Tensor:
  """Binary-lock-in contact reward with distance shaping during approach.

  Three cases, per finger, per frame:
    - ref_flag == 0:                  reward = 0              (don't care)
    - ref_flag == 1 and force > eps:  reward = A              (contact achieved)
    - ref_flag == 1 and force <= eps: reward = exp(-beta * dist)  (approach shaping)

  - ref_flag: preprocessed binary flag, 1 when MANO reference says this finger
    should contact at this frame.
  - force: contact sensor netforce magnitude for the fingertip, read from the
    (hand_site ↔ object_body) ContactSensor.
  - dist: ||robot_tip - ref_contact_point|| where robot_tip is the
    `track_hand_{side}_{finger}_tip` site and ref_contact_point is the per-frame
    target on the object (from preprocessing, transformed into current sim object
    pose).

  Specific-point supervision is provided separately by the `r/l_{finger}_tip`
  tracking reward (exp(-s * ||mano_tip - robot_tip||)); this term only rewards
  contact achievement and shapes the approach when no contact has landed yet.
  """
  command = _cmd(env, command_name)
  si = _side_idx(command, side)
  fi = FINGER_NAMES.index(finger)

  ref_pt = command.ref_contact_pos_w[:, si, fi]  # (B, 3)
  robot_pt = command.robot_tip_pos_w[:, si, fi]  # (B, 3)
  dist = torch.norm(ref_pt - robot_pt, dim=-1)  # (B,)

  sensor: ContactSensor = env.scene[sensor_name]
  force_mag = torch.norm(sensor.data.force[:, fi], dim=-1)  # (B,)
  contacted = force_mag > eps  # (B,) bool

  flag = command.ref_contact_flags[:, si, fi]  # (B,) 0 or 1
  shaping = torch.exp(-beta * dist)  # (B,)
  reward = torch.where(contacted, torch.full_like(shaping, A), shaping)  # (B,)
  return flag * reward


def overforce_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  finger: str,
  threshold: float,
) -> torch.Tensor:
  """Clamp-above penalty on fingertip contact-force magnitude.

  Returns max(||F|| - threshold, 0) per env (B,). Use a negative weight so
  excessive contact force is penalized. Not gated on ref_contact_flags —
  excessive force at a fingertip is a physics red flag regardless of
  whether the reference says "should contact" at that frame.

  Derived from ProtoMotions' contact_force_penalty (SIGGRAPH Asia 2025).
  """
  fi = FINGER_NAMES.index(finger)
  sensor: ContactSensor = env.scene[sensor_name]
  force_mag = torch.norm(sensor.data.force[:, fi], dim=-1)  # (B,)
  return torch.clamp(force_mag - threshold, min=0.0)


def force_magnitude_metric(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  finger: str,
) -> torch.Tensor:
  """Zero-weighted diagnostic: fingertip contact-force magnitude (N).

  Register with weight=0.0 so it appears in Episode_Reward logs without
  affecting learning.
  """
  fi = FINGER_NAMES.index(finger)
  sensor: ContactSensor = env.scene[sensor_name]
  return torch.norm(sensor.data.force[:, fi], dim=-1)


def penetration_depth_metric(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  finger: str,
) -> torch.Tensor:
  """Zero-weighted diagnostic: fingertip penetration depth (m, >=0).

  Reads dist from a ContactSensor with reduce=mindist. Returns
  max(-dist, 0): zero when separated, positive and equal to overlap depth
  when the fingertip is embedded in the object. Register with weight=0.0
  to log without affecting learning.
  """
  fi = FINGER_NAMES.index(finger)
  sensor: ContactSensor = env.scene[sensor_name]
  dist = sensor.data.dist[:, fi]  # (B,) signed; <0 = overlap
  return torch.clamp(-dist, min=0.0)


def contact_force_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  side: str,
  scale: float,
) -> torch.Tensor:
  """ManipTrans contact reward: exp(-scale / (sum_masked_force + eps)).

  Force is masked by fingertip-object distance: only counts when within 2-3cm.
  mask = clamp((0.03 - dist) / 0.01, 0, 1)
  """
  command = _cmd(env, command_name)
  si = _side_idx(command, side)

  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force  # (B, n_primaries, 3)
  force_mag = torch.norm(force, dim=-1)  # (B, n_primaries)

  # Per-finger distance masking using precomputed MANO tip-to-object-surface distance
  # clamp((0.03 - dist) / 0.01, 0, 1)
  tip_dist = command.mano_tips_distance[:, si]  # (B, 5)
  dist_mask = torch.clamp((0.03 - tip_dist) / 0.01, 0.0, 1.0)  # (B, 5)

  masked_force = force_mag * dist_mask  # (B, 5)
  total_force = masked_force.sum(dim=-1)  # (B,)

  return torch.exp(-scale / (total_force + 1e-5))


# --- Per-side object tracking rewards (Stage 2) ---


def obj_pos_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  side: str,
  scale: float,
) -> torch.Tensor:
  """exp(-scale * ||ref_obj_pos - sim_obj_pos||) for one side."""
  command = _cmd(env, command_name)
  si = _side_idx(command, side)
  error = torch.norm(
    command.ref_obj_pos_w[:, si] - command.sim_obj_pos_w[:, si], dim=-1
  )
  return torch.exp(-scale * error)


def obj_rot_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  side: str,
  scale: float,
) -> torch.Tensor:
  """exp(-scale * quat_angle(ref_obj, sim_obj)) for one side."""
  command = _cmd(env, command_name)
  si = _side_idx(command, side)
  error = quat_error_magnitude(
    command.ref_obj_quat_w[:, si], command.sim_obj_quat_w[:, si]
  )
  return torch.exp(-scale * error)


def obj_vel_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  side: str,
  scale: float,
) -> torch.Tensor:
  """exp(-scale * mean(|ref_vel - sim_vel|)) for one side."""
  command = _cmd(env, command_name)
  si = _side_idx(command, side)
  error = torch.mean(
    torch.abs(command.ref_obj_vel_w[:, si] - command.sim_obj_vel_w[:, si]), dim=-1
  )
  return torch.exp(-scale * error)


def obj_angvel_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  side: str,
  scale: float,
) -> torch.Tensor:
  """exp(-scale * mean(|ref_angvel - sim_angvel|)) for one side."""
  command = _cmd(env, command_name)
  si = _side_idx(command, side)
  error = torch.mean(
    torch.abs(command.ref_obj_angvel_w[:, si] - command.sim_obj_angvel_w[:, si]), dim=-1
  )
  return torch.exp(-scale * error)
