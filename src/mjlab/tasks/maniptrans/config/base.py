"""Base task registration for ManipTrans hand types."""

from typing import Callable

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.entity import EntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.maniptrans import mdp as mt_mdp
from mjlab.tasks.maniptrans.maniptrans_env_cfg import make_maniptrans_env_cfg
from mjlab.tasks.maniptrans.mdp import ManipTransCommandCfg
from mjlab.tasks.registry import register_mjlab_task

# ManipTrans per-finger reward scales
_FINGER_SCALES = {
  "thumb": (0.9, 100.0),
  "index": (0.8, 90.0),
  "middle": (0.75, 80.0),
  "ring": (0.6, 60.0),
  "pinky": (0.6, 60.0),
}

# Stratified per-finger contact-match weights (object-interaction only).
_CONTACT_MATCH_WEIGHTS = {
  "thumb": 0.7,
  "index": 0.6,
  "middle": 0.55,
  "ring": 0.45,
  "pinky": 0.45,
}


def _add_per_side_rewards(cfg: ManagerBasedRlEnvCfg, sides: tuple[str, ...]) -> None:
  """Add per-side motion-tracking rewards (hand-only, no object terms)."""
  side_prefixes = {"right": "r", "left": "l"}

  for side in sides:
    p = side_prefixes[side]

    cfg.rewards[f"{p}_wrist_pos"] = RewardTermCfg(
      func=mt_mdp.mano_wrist_pos_error_exp,
      weight=0.1,
      params={"command_name": "motion", "side": side, "scale": 40.0},
    )
    cfg.rewards[f"{p}_wrist_rot"] = RewardTermCfg(
      func=mt_mdp.mano_wrist_rot_error_exp,
      weight=0.6,
      params={"command_name": "motion", "side": side, "scale": 1.0},
    )

    for finger, (weight, scale) in _FINGER_SCALES.items():
      cfg.rewards[f"{p}_{finger}_tip"] = RewardTermCfg(
        func=mt_mdp.mano_fingertip_pos_error_exp,
        weight=weight,
        params={"command_name": "motion", "side": side, "finger": finger, "scale": scale},
      )

    cfg.rewards[f"{p}_level1"] = RewardTermCfg(
      func=mt_mdp.mano_level_pos_error_exp,
      weight=0.5,
      params={"command_name": "motion", "side": side, "level": 1, "scale": 50.0},
    )
    cfg.rewards[f"{p}_level2"] = RewardTermCfg(
      func=mt_mdp.mano_level_pos_error_exp,
      weight=0.3,
      params={"command_name": "motion", "side": side, "level": 2, "scale": 40.0},
    )

    cfg.rewards[f"{p}_power"] = RewardTermCfg(
      func=mt_mdp.power_penalty,
      weight=0.5,
      params={"command_name": "motion", "action_name": "maniptrans", "side": side, "scale": 10.0},
    )
    cfg.rewards[f"{p}_wrist_power"] = RewardTermCfg(
      func=mt_mdp.wrist_power_penalty,
      weight=0.5,
      params={"command_name": "motion", "action_name": "maniptrans", "side": side, "scale": 2.0},
    )

    cfg.rewards[f"{p}_wrist_vel"] = RewardTermCfg(
      func=mt_mdp.mano_wrist_vel_error_exp,
      weight=0.1,
      params={"command_name": "motion", "side": side, "scale": 1.0},
    )
    cfg.rewards[f"{p}_wrist_angvel"] = RewardTermCfg(
      func=mt_mdp.mano_wrist_angvel_error_exp,
      weight=0.05,
      params={"command_name": "motion", "side": side, "scale": 1.0},
    )

    cfg.rewards[f"{p}_joints_vel"] = RewardTermCfg(
      func=mt_mdp.joints_vel_error_exp,
      weight=0.1,
      params={"command_name": "motion", "side": side, "scale": 1.0},
    )


def add_object_interaction_rewards(
  cfg: ManagerBasedRlEnvCfg, sides: tuple[str, ...]
) -> None:
  """Add per-side per-finger contact_match rewards.

  Object/residual trainers call this; the hand-only base trainer does not.
  Requires `{p}_fingertip_penetration` and `{p}_fingertip_contact` sensors
  to exist on the scene (added by the object-aware trainer).
  """
  side_prefixes = {"right": "r", "left": "l"}
  for side in sides:
    p = side_prefixes[side]

    for finger in ("thumb", "index", "middle", "ring", "pinky"):
      cfg.rewards[f"{p}_{finger}_contact_match"] = RewardTermCfg(
        func=mt_mdp.contact_point_match_reward,
        weight=_CONTACT_MATCH_WEIGHTS[finger],
        params={
          "command_name": "motion",
          "sensor_name": f"{p}_fingertip_penetration",
          "side": side,
          "finger": finger,
          "beta": 40.0,
          "gamma": 200.0,
          "tol": 0.002,
        },
      )


def _set_command_params(
  cfg: ManagerBasedRlEnvCfg,
  body_mapping: dict,
  side: str,
) -> None:
  """Set ManipTransCommand params and per-side rewards."""
  sides = ("right", "left") if side == "bimanual" else (side,)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, ManipTransCommandCfg)
  motion_cmd.sides = sides
  motion_cmd.body_mapping = body_mapping

  _add_per_side_rewards(cfg, sides)


def make_hand_env_cfg(
  get_cfg: Callable[[str], EntityCfg],
  body_mapping: dict,
  side: str,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_maniptrans_env_cfg()
  cfg.scene.entities = {"hand": get_cfg(side)}
  _set_command_params(cfg, body_mapping, side)

  if play:
    cfg.scene.num_envs = 4
    cfg.observations["actor"].enable_corruption = False
    cfg.episode_length_s = int(1e9)
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, ManipTransCommandCfg)
    motion_cmd.sampling_mode = "start"

  return cfg


def make_ppo_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 512, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 512, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.0,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=2.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=100,
    num_steps_per_env=32,
    max_iterations=1000,
  )


def register_hand(
  robot_name: str,
  get_cfg: Callable[[str], EntityCfg],
  body_mapping: dict,
) -> None:
  """Register right, left, and bimanual tasks for a hand type."""
  for side in ("right", "left", "bimanual"):
    register_mjlab_task(
      task_id=f"mjlab-maniptrans-{robot_name}-{side}",
      env_cfg=make_hand_env_cfg(get_cfg, body_mapping, side),
      play_env_cfg=make_hand_env_cfg(
        get_cfg, body_mapping, side, play=True
      ),
      rl_cfg=make_ppo_cfg(f"maniptrans_{robot_name}"),
      runner_cls=MjlabOnPolicyRunner,
    )
