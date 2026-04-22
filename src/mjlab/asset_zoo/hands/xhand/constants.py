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

# Body-name mapping consumed by ManipTransCommand. Names are side-agnostic;
# ManipTransCommand constructs `f"{side}_{body_name}"` at lookup time.
BODY_MAPPING = {
  "all": (
    ("hand_thumb_bend_link", "thumb_proximal"),
    ("hand_thumb_rota_link1", "thumb_proximal"),
    ("hand_thumb_rota_link2", "thumb_intermediate"),
    ("hand_index_bend_link", "index_proximal"),
    ("hand_index_rota_link1", "index_proximal"),
    ("hand_index_rota_link2", "index_intermediate"),
    ("hand_mid_link1", "middle_proximal"),
    ("hand_mid_link2", "middle_intermediate"),
    ("hand_ring_link1", "ring_proximal"),
    ("hand_ring_link2", "ring_intermediate"),
    ("hand_pinky_link1", "pinky_proximal"),
    ("hand_pinky_link2", "pinky_intermediate"),
  ),
  "level1": {
    "thumb": "hand_thumb_bend_link",
    "index": "hand_index_bend_link",
    "middle": "hand_mid_link1",
    "ring": "hand_ring_link1",
    "pinky": "hand_pinky_link1",
  },
  "level2": {
    "thumb": "hand_thumb_rota_link2",
    "index": "hand_index_rota_link2",
    "middle": "hand_mid_link2",
    "ring": "hand_ring_link2",
    "pinky": "hand_pinky_link2",
  },
}


def get_cfg(side: str) -> EntityCfg:
  return get_hand_cfg(XMLS, side)
