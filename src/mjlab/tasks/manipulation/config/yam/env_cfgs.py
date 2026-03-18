from typing import Any, Literal

import mujoco

from mjlab.asset_zoo.robots import (
  YAM_ACTION_SCALE,
  get_yam_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.utils.noise import RgbAugmentationCfg


def get_cube_spec(cube_size: float = 0.02, mass: float = 0.05) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=(0.8, 0.2, 0.2, 1.0),
  )
  return spec


def yam_lift_cube_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg()

  cfg.scene.entities = {
    "robot": get_yam_robot_cfg(),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = YAM_ACTION_SCALE

  gripper_joints = ("left_finger", "right_finger")
  cfg.observations["actor"].terms["gripper_pos"].params[
    "asset_cfg"
  ].joint_names = gripper_joints
  cfg.observations["actor"].terms["gripper_vel"].params[
    "asset_cfg"
  ].joint_names = gripper_joints
  cfg.observations["critic"].terms["gripper_pos"].params[
    "asset_cfg"
  ].joint_names = gripper_joints
  cfg.observations["critic"].terms["gripper_vel"].params[
    "asset_cfg"
  ].joint_names = gripper_joints

  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    "grasp_site",
  )
  cfg.observations["critic"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    "grasp_site",
  )
  cfg.rewards["lift"].params["asset_cfg"].site_names = ("grasp_site",)

  fingertip_geoms = r"tip_[lr]_\d+_collision"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  # Configure collision sensor pattern.
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link_6"

  cfg.viewer.body_name = "arm"

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

    # Higher command resampling frequency for more dynamic play.
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)

  return cfg


def yam_lift_cube_vision_env_cfg(
  cam_type: Literal["rgb", "depth"],
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = yam_lift_cube_env_cfg(play=play)

  camera_names = ["robot/camera_d405"]
  cam_kwargs = {
    "robot/camera_d405": {
      "height": 32,
      "width": 32,
    },
  }
  shared_cam_kwargs = dict(
    data_types=(cam_type,),
    enabled_geom_groups=(0, 3),
    use_shadows=True,
    use_textures=True,
  )

  cam_terms = {}
  for cam_name in camera_names:
    cam_cfg = CameraSensorCfg(
      name=cam_name.split("/")[-1],
      camera_name=cam_name,
      **cam_kwargs[cam_name],  # type: ignore[invalid-argument-type]
      **shared_cam_kwargs,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)
    param_kwargs: dict[str, Any] = {"sensor_name": cam_cfg.name}
    if cam_type == "depth":
      param_kwargs["cutoff_distance"] = 0.5
      func = manipulation_mdp.camera_depth
    else:
      func = manipulation_mdp.camera_rgb
    noise = None
    if cam_type == "rgb":
      noise = RgbAugmentationCfg(
        max_shift_pixels=2,
        brightness=0.15,
        contrast=0.0,
        saturation=0.0,
        color_scale_range=(1.0, 1.0),
        blur_kernel_size=3,
        blur_sigma_range=(0.1, 1.0),
      )
    cam_terms[f"{cam_name.split('/')[-1]}_{cam_type}"] = ObservationTermCfg(
      func=func, params=param_kwargs, noise=noise
    )

  camera_obs = ObservationGroupCfg(
    terms=cam_terms, enable_corruption=not play, concatenate_terms=True
  )
  cfg.observations["camera"] = camera_obs

  cfg.events["camera_focal"] = EventTermCfg(
    func=dr.cam_intrinsic,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", camera_names=("camera_d405",)),
      "operation": "scale",
      "distribution": "uniform",
      "axes": [0, 1],
      "ranges": (0.95, 1.05),
    },
  )
  cfg.events["camera_principal"] = EventTermCfg(
    func=dr.cam_intrinsic,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", camera_names=("camera_d405",)),
      "operation": "add",
      "distribution": "uniform",
      "axes": [2, 3],
      "ranges": (-0.01, 0.01),
    },
  )
  cfg.events["camera_pos"] = EventTermCfg(
    func=dr.cam_pos,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", camera_names=("camera_d405",)),
      "operation": "add",
      "distribution": "uniform",
      "ranges": (-0.002, 0.002),
    },
  )
  cfg.events["camera_quat"] = EventTermCfg(
    func=dr.cam_quat,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", camera_names=("camera_d405",)),
      "roll_range": (-0.017, 0.017),
      "pitch_range": (-0.017, 0.017),
      "yaw_range": (-0.017, 0.017),
    },
  )

  if cam_type == "rgb":
    # Disable terrain texture so plane color is controlled by geom_rgba.
    assert isinstance(cfg.scene.terrain, TerrainEntityCfg)
    cfg.scene.terrain.textures = ()
    cfg.scene.terrain.materials = ()

    cfg.events["cube_color"] = EventTermCfg(
      func=dr.geom_rgba,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("cube", geom_names=(".*",)),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0, 1, 2],
        "ranges": (0.0, 1.0),
      },
    )
    cfg.events["plane_color"] = EventTermCfg(
      func=dr.geom_rgba,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("terrain", geom_names=("terrain",)),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0, 1, 2],
        "ranges": (0.3, 0.7),
      },
    )
    cfg.events["gripper_color"] = EventTermCfg(
      func=dr.mat_rgba,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", material_names=("finger",)),
        "operation": "add",
        "distribution": "uniform",
        "axes": [0, 1, 2],
        "ranges": (-0.1, 0.15),
      },
    )
    cfg.events["light_pos"] = EventTermCfg(
      func=dr.light_pos,
      mode="reset",
      params={
        "operation": "add",
        "distribution": "uniform",
        "ranges": (-1.0, 1.0),
      },
    )
    cfg.events["light_dir"] = EventTermCfg(
      func=dr.light_dir,
      mode="reset",
      params={
        "operation": "add",
        "distribution": "uniform",
        "ranges": (-0.5, 0.5),
      },
    )

  # Pop privileged info from actor observations.
  actor_obs = cfg.observations["actor"]
  actor_obs.terms.pop("ee_to_cube")
  actor_obs.terms.pop("cube_to_goal")

  # Add goal_position to actor observations.
  actor_obs.terms["goal_position"] = ObservationTermCfg(
    func=manipulation_mdp.target_position,
    params={
      "command_name": "lift_height",
      "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
    },
    # NOTE: No noise for goal position.
  )

  return cfg
