from mjlab.asset_zoo.hands.inspire.constants import BODY_MAPPING, get_cfg
from mjlab.tasks.maniptrans.config.base import register_hand

register_hand("inspire", get_cfg, BODY_MAPPING)
