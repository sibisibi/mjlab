"""Inspire hand constants."""

from paths import HANDS_DIR
from mjlab.asset_zoo.hands.base import get_hand_cfg, xmls_from_dir
from mjlab.entity import EntityCfg

XMLS = xmls_from_dir(HANDS_DIR / "inspire")

# Body-name mapping consumed by ManipTransCommand. Names are side-agnostic;
# ManipTransCommand constructs `f"{side}_{body_name}"` at lookup time.
BODY_MAPPING = {
  "all": (
    ("thumb_proximal_base", "thumb_proximal"),
    ("thumb_proximal", "thumb_proximal"),
    ("thumb_intermediate", "thumb_intermediate"),
    ("thumb_distal", "thumb_distal"),
    ("index_proximal", "index_proximal"),
    ("index_intermediate", "index_intermediate"),
    ("middle_proximal", "middle_proximal"),
    ("middle_intermediate", "middle_intermediate"),
    ("ring_proximal", "ring_proximal"),
    ("ring_intermediate", "ring_intermediate"),
    ("pinky_proximal", "pinky_proximal"),
    ("pinky_intermediate", "pinky_intermediate"),
  ),
  "level1": {
    "thumb": "thumb_proximal",
    "index": "index_proximal",
    "middle": "middle_proximal",
    "ring": "ring_proximal",
    "pinky": "pinky_proximal",
  },
  "level2": {
    "thumb": "thumb_intermediate",
    "index": "index_intermediate",
    "middle": "middle_intermediate",
    "ring": "ring_intermediate",
    "pinky": "pinky_intermediate",
  },
}


def get_cfg(side: str) -> EntityCfg:
  return get_hand_cfg(XMLS, side)
