"""ManipTrans action term: trajectory-residual wrist + scaled finger control.

Action dim is always `n_dofs` (36 for bimanual xhand). The policy's action
tensor IS the applied action that gets turned into joint targets below:
  wrist_target = ref_wrist + wrist_action * wrist_residual_scale
  finger_target = scaled/clamped from action via finger_residual_scale

Stage 2 residual policy composition (`applied = base + residual * scale`) is
handled INSIDE `rl/residual_actor.py::ResidualActor`, NOT here — the action
tensor arriving at this term is already the applied action. This term doesn't
need to know whether it's Stage 1 or Stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from .commands import ManipTransCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ManipTransActionCfg(ActionTermCfg):
  """Configuration for ManipTrans trajectory-residual action term."""

  entity_name: str
  command_name: str
  wrist_actuator_names: tuple[str, ...] = (".*forearm.*|.*pos_[xyz].*",)
  finger_actuator_names: tuple[str, ...] = (".*hand.*|.*finger.*|.*thumb.*|.*index.*|.*mid.*|.*ring.*|.*pinky.*",)
  wrist_residual_scale: float = 0.05
  finger_residual_scale: float = 1.0

  def build(self, env: ManagerBasedRlEnv) -> ManipTransAction:
    return ManipTransAction(self, env)


class ManipTransAction(ActionTerm):
  """Trajectory-residual action for ManipTrans. action_dim = n_dofs."""

  cfg: ManipTransActionCfg
  _entity: Entity

  def __init__(self, cfg: ManipTransActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self._command_name = cfg.command_name

    wrist_ids, wrist_names = self._entity.find_joints_by_actuator_names(
      cfg.wrist_actuator_names
    )
    finger_ids, finger_names = self._entity.find_joints_by_actuator_names(
      cfg.finger_actuator_names
    )

    self._wrist_ids = torch.tensor(wrist_ids, device=self.device, dtype=torch.long)
    self._finger_ids = torch.tensor(finger_ids, device=self.device, dtype=torch.long)
    self._wrist_names = wrist_names
    self._finger_names = finger_names

    self._n_wrist = len(wrist_ids)
    self._n_finger = len(finger_ids)
    self._n_dofs = self._n_wrist + self._n_finger
    self._action_dim = self._n_dofs

    limits = self._entity.data.soft_joint_pos_limits[0]
    self._finger_lower = limits[self._finger_ids, 0]
    self._finger_upper = limits[self._finger_ids, 1]
    self._finger_range = self._finger_upper - self._finger_lower

    self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
    self._all_joint_ids = torch.cat([self._wrist_ids, self._finger_ids])

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def n_dofs(self) -> int:
    """Number of applied action dimensions (hand DoFs). Always n_dofs — Stage 2
    residual composition happens inside the ResidualActor, not here."""
    return self._n_dofs

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions

  def apply_actions(self) -> None:
    command = cast(
      ManipTransCommand, self._env.command_manager.get_term(self._command_name)
    )
    ref_joint_pos = command.ref_joint_pos  # (B, n_dofs)

    action = self._raw_actions
    wrist_action = action[:, :self._n_wrist]
    finger_action = action[:, self._n_wrist:]

    # Wrist: ref + action * scale
    ref_wrist = ref_joint_pos[:, self._wrist_ids]
    wrist_target = ref_wrist + wrist_action * self.cfg.wrist_residual_scale

    # Fingers
    ref_finger = ref_joint_pos[:, self._finger_ids]
    if self.cfg.finger_residual_scale >= 1.0:
      finger_action_clamped = torch.clamp(finger_action, -1.0, 1.0)
      finger_target = (
        0.5 * (finger_action_clamped + 1.0) * self._finger_range + self._finger_lower
      )
    else:
      finger_target = (
        ref_finger + finger_action * self.cfg.finger_residual_scale * self._finger_range
      )

    finger_target = torch.clamp(finger_target, self._finger_lower, self._finger_upper)

    target = torch.cat([wrist_target, finger_target], dim=-1)
    self._entity.set_joint_position_target(target, joint_ids=self._all_joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0
