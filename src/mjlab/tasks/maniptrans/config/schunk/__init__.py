from mjlab.asset_zoo.hands.schunk.constants import ROOT_BODIES, get_cfg
from mjlab.tasks.maniptrans.config.base import register_hand

register_hand("schunk", get_cfg, ROOT_BODIES)
