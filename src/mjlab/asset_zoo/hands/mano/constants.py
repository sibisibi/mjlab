"""Mano hand constants."""

from path import ROBOT_DIR
from mjlab.asset_zoo.hands.base import get_hand_cfg, xmls_from_dir
from mjlab.entity import EntityCfg

XMLS = xmls_from_dir(ROBOT_DIR / "mano")

ROOT_BODIES = {
  "right": "right_palm",
  "left": "left_palm",
  "bimanual": "right_palm",
}


def get_cfg(side: str) -> EntityCfg:
  return get_hand_cfg(XMLS, side)
