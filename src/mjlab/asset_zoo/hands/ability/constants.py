"""Ability hand constants."""

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.hands.base import get_hand_cfg, xmls_from_dir
from mjlab.entity import EntityCfg

XMLS = xmls_from_dir(MJLAB_SRC_PATH / "asset_zoo" / "hands" / "ability")

ROOT_BODIES = {
  "right": "L_forearm_ty_link",
  "left": "R_forearm_ty_link",
  "bimanual": "L_forearm_ty_link",
}


def get_cfg(side: str) -> EntityCfg:
  return get_hand_cfg(XMLS, side)
