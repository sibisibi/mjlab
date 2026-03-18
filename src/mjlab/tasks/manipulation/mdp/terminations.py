from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def object_out_of_bounds(
  env: ManagerBasedRlEnv,
  object_name: str,
  bounds_min: tuple[float, float],
  bounds_max: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate if an object leaves an xy bounding box in the robot base frame."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  obj_pos_w = obj.data.root_link_pos_w
  base_pos_w = robot.data.root_link_pos_w
  base_quat_w = robot.data.root_link_quat_w
  obj_pos_b = quat_apply(quat_inv(base_quat_w), obj_pos_w - base_pos_w)
  obj_xy = obj_pos_b[:, :2]
  lo = torch.tensor(bounds_min, device=obj_xy.device)
  hi = torch.tensor(bounds_max, device=obj_xy.device)
  return (obj_xy < lo).any(dim=-1) | (obj_xy > hi).any(dim=-1)
