from mjlab.asset_zoo.hands.xhand.constants import ROOT_BODIES, get_cfg
from mjlab.tasks.maniptrans.config.base import register_hand

register_hand("xhand", get_cfg, ROOT_BODIES)
