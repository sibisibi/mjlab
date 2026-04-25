"""ManipTrans environment configuration for mjlab."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import observations as mdp_obs
from mjlab.envs.mdp import rewards as mdp_rewards
from mjlab.envs.mdp import terminations as mdp_term
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.maniptrans import mdp as mt_mdp
from mjlab.tasks.maniptrans.mdp import ManipTransActionCfg, ManipTransCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig


def make_maniptrans_env_cfg() -> ManagerBasedRlEnvCfg:
  """Base ManipTrans dexterous manipulation environment.

  Returns a config with no entities set. The per-robot config
  (e.g. config/base.py) must set scene.entities, command params afterwards.
  """

  # --- Observations ---

  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mt_mdp.hand_joint_pos,
      params={"asset_cfg": SceneEntityCfg("hand", joint_names=(".*",))},
    ),
    "joint_cos_sin": ObservationTermCfg(
      func=mt_mdp.hand_joint_cos_sin,
      params={"asset_cfg": SceneEntityCfg("hand", joint_names=(".*",))},
    ),
    "wrist_state": ObservationTermCfg(
      func=mt_mdp.wrist_state,
      params={"asset_cfg": SceneEntityCfg("hand")},
    ),
    "dq": ObservationTermCfg(
      func=mt_mdp.hand_joint_vel,
      params={"asset_cfg": SceneEntityCfg("hand", joint_names=(".*",))},
    ),
    # Stage-invariant: first n_dofs of raw action. In Stage 1 this equals the full
    # action; in Stage 2 it's the first-half slot. Keeps `last_action` obs width
    # constant at n_dofs across stages so Stage 1 checkpoints' obs prefixes align
    # when loaded as a frozen base in Stage 2 residual training.
    "last_action": ObservationTermCfg(func=mt_mdp.last_applied_action),
    # MANO tracking — absolute references
    "mano_wrist_vel": ObservationTermCfg(
      func=mt_mdp.mano_wrist_vel,
      params={"command_name": "motion"},
    ),
    "mano_wrist_quat": ObservationTermCfg(
      func=mt_mdp.mano_wrist_quat,
      params={"command_name": "motion"},
    ),
    "mano_wrist_angvel": ObservationTermCfg(
      func=mt_mdp.mano_wrist_angvel,
      params={"command_name": "motion"},
    ),
    # MANO tracking — deltas
    "mano_wrist_pos_delta": ObservationTermCfg(
      func=mt_mdp.mano_wrist_pos_delta,
      params={"command_name": "motion"},
    ),
    "mano_wrist_rot_delta": ObservationTermCfg(
      func=mt_mdp.mano_wrist_rot_delta,
      params={"command_name": "motion"},
    ),
    "mano_fingertip_pos_delta": ObservationTermCfg(
      func=mt_mdp.mano_fingertip_pos_delta,
      params={"command_name": "motion"},
    ),
    "mano_wrist_vel_delta": ObservationTermCfg(
      func=mt_mdp.mano_wrist_vel_delta,
      params={"command_name": "motion"},
    ),
    "mano_wrist_angvel_delta": ObservationTermCfg(
      func=mt_mdp.mano_wrist_angvel_delta,
      params={"command_name": "motion"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
  }

  # --- Actions ---

  actions: dict[str, ActionTermCfg] = {
    "maniptrans": ManipTransActionCfg(
      entity_name="hand",
      command_name="motion",
      wrist_actuator_names=(".*forearm.*|.*pos_[xyz].*|.*rot_[xyz].*",),
      finger_actuator_names=(".*thumb.*", ".*index.*", ".*mid.*", ".*ring.*", ".*pinky.*"),
      wrist_residual_scale=0.05,  # Override via CLI
      finger_residual_scale=1.0,  # Override via CLI
    ),
  }

  # --- Commands ---

  commands: dict[str, CommandTermCfg] = {
    "motion": ManipTransCommandCfg(
      resampling_time_range=(1.0e9, 1.0e9),
      entity_name="hand",
      motion_file="",  # Set per-hand at runtime
      sides=(),  # Set per-hand in config/base.py
      body_mapping={},  # Set per-hand in config/base.py
    ),
  }

  # --- Rewards ---
  # Per-side rewards are added in config/base.py where sides are known.
  # Only side-agnostic rewards here.

  rewards: dict[str, RewardTermCfg] = {
    "action_rate": RewardTermCfg(
      func=mdp_rewards.action_rate_l2,
      weight=-0.01,
    ),
    "joint_limits": RewardTermCfg(
      func=mdp_rewards.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("hand", joint_names=(".*",))},
    ),
  }

  # --- Terminations ---

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(
      func=mdp_term.time_out,
      time_out=True,
    ),
    "velocity_diverged": TerminationTermCfg(
      func=mt_mdp.velocity_diverged,
      params={
        "max_lin_vel": 100.0,
        "max_ang_vel": 200.0,
        "asset_cfg": SceneEntityCfg("hand"),
      },
    ),
    "fingertip_diverged": TerminationTermCfg(
      func=mt_mdp.fingertip_diverged,
      params={"command_name": "motion", "threshold": 0.3, "grace_steps": 15},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=4096,
      env_spacing=0.5,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="hand",
      body_name="",  # Set per-robot.
      distance=0.5,
      elevation=-20.0,
      azimuth=140.0,
    ),
    sim=SimulationCfg(
      nconmax=200,
      njmax=600,
      mujoco=MujocoCfg(
        timestep=0.002,
        iterations=20,
        ls_iterations=50,
        integrator="implicitfast",
        cone="elliptic",
        o_solimp=(0.95, 0.99, 0.002, 0.5, 2.0),
      ),
    ),
    decimation=6,
    episode_length_s=20.0,
  )
