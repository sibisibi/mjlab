"""Base hand constants shared across all dexterous hand types."""

from pathlib import Path

import mujoco

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

ACTUATOR = XmlActuatorCfg(
  target_names_expr=(".*",),
)

COLLISION = CollisionCfg(
  geom_names_expr=("collision_hand_.*",),
  contype=1,
  conaffinity=0,
  condim=3,
  friction=(1.0, 0.005, 0.0001),
  disable_other_geoms=True,
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(ACTUATOR,),
)


def xmls_from_dir(asset_dir: Path) -> dict[str, Path]:
  """Standard XML paths for a hand: right, left, bimanual."""
  return {
    "right": asset_dir / "right.xml",
    "left": asset_dir / "left.xml",
    "bimanual": asset_dir / "bimanual.xml",
  }


def get_hand_cfg(xmls: dict[str, Path], side: str) -> EntityCfg:
  """Create an EntityCfg for a dexterous hand."""
  xml_path = str(xmls[side])
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      joint_pos={".*": 0.0},
    ),
    spec_fn=lambda p=xml_path: mujoco.MjSpec.from_file(p),
    articulation=ARTICULATION,
    collisions=(COLLISION,),
  )
