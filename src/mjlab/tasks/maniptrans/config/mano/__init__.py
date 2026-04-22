from mjlab.asset_zoo.hands.mano.constants import ROOT_BODIES, get_cfg
from mjlab.tasks.maniptrans.config.base import register_hand

register_hand("mano", get_cfg, ROOT_BODIES)
