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


def nan_guard(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Terminate envs whose simulation state contains NaN/Inf.

  Catches what `velocity_sanity` cannot: PyTorch comparison operators return
  False on NaN, so velocity caps silently fail. This explicit isnan check
  fires on the next step's termination cycle, after which `_resample_command`
  resets the env to a clean state.
  """
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  hand: Entity = env.scene[asset_cfg.name]
  bad_joint = (
    torch.isnan(hand.data.joint_pos).any(dim=-1)
    | torch.isnan(hand.data.joint_vel).any(dim=-1)
  )
  # Object state shape (B, n_sides, ...): collapse trailing dims.
  obj_pos = command.sim_obj_pos_w
  obj_vel = command.sim_obj_vel_w
  bad_obj = (
    torch.isnan(obj_pos).flatten(1).any(dim=-1)
    | torch.isnan(obj_vel).flatten(1).any(dim=-1)
  )
  return bad_joint | bad_obj


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
  found_history_key: str,
  side: str,
  grace_steps: int,
) -> torch.Tensor:
  """Terminate when MANO expects contact but the robot hasn't made any
  in the recent history window.

  Gate: `ref_contact_flags == 1` (preprocessed MANO binary predicate — same
  gate the `contact_point_match_reward` uses). The policy is killed for
  failing the exact predicate the reward shapes against.

  Signal: `found_history > 0` across all slots (narrowphase contact flag
  from the penetration sensor, written by `contact_found_history`). History
  length is set where the obs term is wired; the window determines how many
  consecutive frames of missing contact are tolerated.
  """
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  si = command._side_list.index(side)
  ref_flag = command.ref_contact_flags[:, si]  # (B, 5), 0 or 1

  buf = env.extras[found_history_key]  # (B, history_len, 5) of 0/1 floats
  has_any_contact = (buf > 0).any(dim=1)  # (B, 5)

  missing = (ref_flag > 0.5) & ~has_any_contact  # (B, 5)
  exceeded = torch.any(missing, dim=-1)  # (B,)

  return exceeded & (env.episode_length_buf >= grace_steps)


def contact_missed_too_long(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold_steps: int,
  grace_steps: int,
) -> torch.Tensor:
  """Terminate if ANY finger's consecutive-miss streak reaches threshold_steps,
  skipped during the first grace_steps of each episode.

  The streak counter (maintained on ManipTransCommand.contact_miss_counter,
  shape (B, n_sides, 5)) increments each step where ref_flag==1 AND found==0
  for that finger, and resets to 0 when ref_flag==0 OR found==1. Each of the
  10 (bimanual) or 5 (single-hand) fingers tracked independently. Counter
  itself keeps ticking during grace; only the kill is suppressed — so the
  contact_miss_max_<p>_<finger> metrics stay representative.

  Set threshold_steps very high (e.g. 999_999) to disable while observing
  contact_miss_max_<p>_<finger> metrics to pick a threshold.
  """
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  exceeded = (command.contact_miss_counter >= threshold_steps).flatten(1).any(dim=-1)
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


def joint_vel_sanity(env: ManagerBasedRlEnv, max_joint_vel: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  hand: Entity = env.scene[asset_cfg.name]
  return torch.norm(hand.data.joint_vel, dim=-1) > max_joint_vel


def obj_lin_vel_sanity(env: ManagerBasedRlEnv, command_name: str, max_obj_vel: float) -> torch.Tensor:
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  return torch.any(torch.norm(command.sim_obj_vel_w, dim=-1) > max_obj_vel, dim=-1)


def obj_ang_vel_sanity(env: ManagerBasedRlEnv, command_name: str, max_obj_angvel: float) -> torch.Tensor:
  command = cast(ManipTransCommand, env.command_manager.get_term(command_name))
  return torch.any(torch.norm(command.sim_obj_angvel_w, dim=-1) > max_obj_angvel, dim=-1)


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


