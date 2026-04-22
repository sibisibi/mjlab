from mjlab.asset_zoo.hands.xhand.constants import BODY_MAPPING, ROOT_BODIES, get_cfg
from mjlab.tasks.maniptrans.config.base import register_hand

register_hand("xhand", get_cfg, ROOT_BODIES, BODY_MAPPING)
