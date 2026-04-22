"""XHand constants."""

from path import ROBOT_DIR
from mjlab.asset_zoo.hands.base import get_hand_cfg, xmls_from_dir
from mjlab.entity import EntityCfg

XMLS = xmls_from_dir(ROBOT_DIR / "xhand")

ROOT_BODIES = {
  "right": "R_forearm_ty_link",
  "left": "L_forearm_ty_link",
  "bimanual": "R_forearm_ty_link",
}


def get_cfg(side: str) -> EntityCfg:
  return get_hand_cfg(XMLS, side)
