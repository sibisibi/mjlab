"""Termination terms for ManipTrans dexterous manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import ManipTransCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def velocity_diverged(
  env: ManagerBasedRlEnv,
  max_lin_vel: float,
  max_ang_vel: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Terminate if wrist velocity is unreasonably high."""
  hand: Entity = env.scene[asset_cfg.name]
  lin = torch.norm(hand.data.root_link_lin_vel_w, dim=-1)
  ang = torch.norm(hand.data.root_link_ang_vel_w, dim=-1)
  return (lin > max_lin_vel) | (ang > max_ang_vel)


def fingertip_diverged(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  grace_steps: int,
) -> torch.Tensor:
  """Terminate if ANY fingertip error exceeds threshold (any side).
  Skipped during first grace_steps of each episode."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  error = torch.norm(
    command.mano_tip_pos_w - command.robot_tip_pos_w, dim=-1
  )  # (B, n_sides, 5)
  exceeded = torch.any(error.reshape(error.shape[0], -1) > threshold, dim=-1)
  return exceeded & (env.episode_length_buf >= grace_steps)


def obj_pos_diverged(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  grace_steps: int,
) -> torch.Tensor:
  """Terminate if object position error exceeds threshold (any side)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  error = torch.norm(
    command.ref_obj_pos_w - command.sim_obj_pos_w, dim=-1
  )  # (B, n_sides)
  exceeded = torch.any(error > threshold, dim=-1)
  return exceeded & (env.episode_length_buf >= grace_steps)


def obj_rot_diverged(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold_deg: float,
  grace_steps: int,
) -> torch.Tensor:
  """Terminate if object rotation error exceeds threshold (any side)."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  error_rad = quat_error_magnitude(
    command.ref_obj_quat_w, command.sim_obj_quat_w
  )  # (B, n_sides)
  threshold_rad = threshold_deg * 3.14159265 / 180.0
  exceeded = torch.any(error_rad > threshold_rad, dim=-1)
  return exceeded & (env.episode_length_buf >= grace_steps)


def contact_expected_but_missing(
  env: ManagerBasedRlEnv,
  command_name: str,
  force_history_key: str,
  side: str,
  dist_threshold: float,
  grace_steps: int,
) -> torch.Tensor:
  """Terminate if fingertips close to object but no contact in recent history.

  ManipTrans rule: `(tips_distance < 0.005) & ~any(contact_history) → terminate`.
  `tips_distance` is the **preprocessed MANO reference** tip-to-object-surface
  distance, not sim state — threshold is in reference space.

  Contact history source is now the 4D `contact_force_history` buffer with
  shape `(B, history_len, n_primaries, 4)` where channel 3 is the force
  magnitude. "Any contact in window" is derived as `magnitude_channel > 0`
  over the history axis. No separate bool history is needed.
  """
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  si = command._side_list.index(side)

  # Preprocessed MANO tip-to-object-surface distance (reference, not sim)
  tip_dist = command.mano_tips_distance[:, si]  # (B, 5)
  tips_close = tip_dist < dist_threshold  # (B, 5)

  # Force history: (B, history_len, 5, 4). Threshold magnitude channel.
  buf = env.extras.get(force_history_key)
  if buf is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  # buf[..., 3] = force magnitude; "was there contact" = any nonzero step.
  has_any_contact = (buf[..., 3] > 0).any(dim=1)  # (B, 5)

  close_no_contact = tips_close & ~has_any_contact  # (B, 5)
  exceeded = torch.any(close_no_contact, dim=-1)  # (B,)

  return exceeded & (env.episode_length_buf >= grace_steps)


def velocity_sanity(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_obj_vel: float,
  max_obj_angvel: float,
  max_joint_vel: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Terminate if object or joint velocities are unreasonably high."""
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  hand: Entity = env.scene[asset_cfg.name]

  # Joint velocity check
  joint_vel = torch.norm(hand.data.joint_vel, dim=-1)
  joint_bad = joint_vel > max_joint_vel

  # Object velocity checks
  obj_vel = torch.norm(command.sim_obj_vel_w, dim=-1)  # (B, n_sides)
  obj_angvel = torch.norm(command.sim_obj_angvel_w, dim=-1)  # (B, n_sides)
  obj_bad = torch.any(obj_vel > max_obj_vel, dim=-1) | torch.any(obj_angvel > max_obj_angvel, dim=-1)

  return joint_bad | obj_bad


def dof_vel_sanity(
  env: ManagerBasedRlEnv,
  max_dof_vel: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Terminate if mean absolute DOF velocity exceeds threshold.

  ManipTrans: torch.abs(current_dof_vel).mean(-1) > 200
  """
  hand: Entity = env.scene[asset_cfg.name]
  return hand.data.joint_vel.abs().mean(dim=-1) > max_dof_vel


