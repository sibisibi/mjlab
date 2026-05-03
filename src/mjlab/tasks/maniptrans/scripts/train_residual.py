"""Residual training for ManipTrans.

Loads frozen Stage 1 base checkpoint(s) and trains only a residual head on top.
Single-hand mode: 1 base + 1 residual MLP. Bimanual two-base mode: 2 per-side
bases + bimanual residual MLP that splits the bimanual obs along per-term
boundaries and recombines wrist-then-finger per side.

Augmentation over the registered hand-only base task: object entity (free),
per-finger contact_match rewards, contact sensors, tactile obs, object obs in
actor + critic, object tracking rewards, object termination criteria, actor
obs filtering (no hand_obj_distance / gt_tips_distance in actor).
"""

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import wandb

import mjlab.tasks.maniptrans.config  # noqa: F401  registers tasks
from mjlab.asset_zoo.objects.entity import get_object_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.maniptrans import mdp as mt_mdp
from mjlab.tasks.maniptrans.config.base import add_object_interaction_rewards
from mjlab.tasks.maniptrans.mdp import ManipTransCommandCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends


CONTACT_MATCH_GAMMA = 200.0
CONTACT_MATCH_TOL = 0.002

OBJECT_REWARD_TERMS = (
  ("obj_pos", mt_mdp.obj_pos_error_exp, 5.0, 80.0),
  ("obj_rot", mt_mdp.obj_rot_error_exp, 1.0, 3.0),
  ("obj_vel", mt_mdp.obj_vel_error_exp, 0.1, 1.0),
  ("obj_angvel", mt_mdp.obj_angvel_error_exp, 0.1, 1.0),
)

# Hand-obs terms whose bimanual dim equals their per-side dim because they
# read entity-root state once for the whole bimanual hand entity.
GLOBAL_HAND_OBS_TERMS = {"wrist_state"}


def _read_base_dims(path: str) -> tuple[int, int]:
  ckpt = torch.load(path, map_location="cpu", weights_only=False)
  sd = ckpt["actor_state_dict"]
  obs_dim = int(sd["obs_normalizer._mean"].shape[-1])
  last_idx = max(int(k.split(".")[1]) for k in sd if k.startswith("mlp."))
  act_dim = int(sd[f"mlp.{last_idx}.bias"].shape[0])
  return obs_dim, act_dim


def build_env_cfg(args):
  task_id = f"mjlab-maniptrans-{args.robot}-{args.side}"
  cfg = load_env_cfg(task_id)
  cfg.scene.num_envs = args.num_envs

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, ManipTransCommandCfg)
  motion_cmd.motion_file = args.motion_file
  motion_cmd.motion_index = args.index
  motion_cmd.pin_objects = False

  sides = ("right", "left") if args.side == "bimanual" else (args.side,)
  add_object_interaction_rewards(cfg, sides)

  # Read object mesh paths + scales from the packed .pt header. The packer
  # filtered to a single object_id, so these are constant across motions.
  packed = torch.load(args.motion_file, weights_only=False)
  pool_rel = packed.get("pool_rel_dir")
  if pool_rel is None:
    raise ValueError(
      f"Packed motion file {args.motion_file!r} has no `pool_rel_dir` "
      f"embedded -- regenerate via package_motion_batch.py."
    )
  pool_dir = Path(args.input_dir) / pool_rel
  right_mesh_rel = packed.get("right_object_mesh_dir")
  left_mesh_rel = packed.get("left_object_mesh_dir")
  if right_mesh_rel is None and left_mesh_rel is None:
    raise ValueError(
      f"Packed motion file {args.motion_file!r} has neither "
      f"right_object_mesh_dir nor left_object_mesh_dir embedded."
    )
  right_obj_dir = str(pool_dir / right_mesh_rel) if right_mesh_rel else None
  left_obj_dir = str(pool_dir / left_mesh_rel) if left_mesh_rel else None
  right_mesh_scale = float(packed.get("right_object_mesh_scale", 1.0))
  left_mesh_scale = float(packed.get("left_object_mesh_scale", 1.0))

  # Bimanual-on-same-item trajectories: build one entity, point both sides at it,
  # else two entities drift between pin snaps and render as twin ghosts.
  shared_object = (
    left_obj_dir is not None and right_obj_dir is not None
    and right_obj_dir == left_obj_dir
  )
  if right_obj_dir is not None:
    cfg.scene.entities["object_right"] = get_object_cfg(
      right_obj_dir, "obj_right", density=args.obj_density, mesh_scale=right_mesh_scale,
    )
  if left_obj_dir is not None and not shared_object:
    cfg.scene.entities["object_left"] = get_object_cfg(
      left_obj_dir, "obj_left", density=args.obj_density, mesh_scale=left_mesh_scale,
    )
  if args.side == "right":
    if "object_left" in cfg.scene.entities:
      del cfg.scene.entities["object_left"]
    motion_cmd.object_entity_names = {"right": "object_right"}
  elif args.side == "left":
    if left_obj_dir is None and not shared_object:
      raise ValueError(
        "--side left requested but trajectory has no left object "
        "(task_info['left_object_mesh_dir'] is null)."
      )
    if shared_object:
      motion_cmd.object_entity_names = {"left": "object_right"}
    else:
      del cfg.scene.entities["object_right"]
      motion_cmd.object_entity_names = {"left": "object_left"}
  else:
    if left_obj_dir is None:
      raise ValueError(
        "--side bimanual requested but trajectory has no left object "
        "(task_info['left_object_mesh_dir'] is null)."
      )
    if shared_object:
      motion_cmd.object_entity_names = {"right": "object_right", "left": "object_right"}
    else:
      motion_cmd.object_entity_names = {"right": "object_right", "left": "object_left"}

  for key, term in cfg.rewards.items():
    if key.endswith("_contact_match"):
      term.weight *= args.contact_match_weight
      term.params["beta"] = args.contact_match_beta
      term.params["gamma"] = CONTACT_MATCH_GAMMA
      term.params["tol"] = CONTACT_MATCH_TOL

  side_pref = {"right": "r", "left": "l"}
  sensors: list[ContactSensorCfg] = []
  for side in sides:
    p = side_pref[side]
    obj_entity = "object_right" if (side == "right" or shared_object) else "object_left"
    obj_pattern = "obj_right" if (side == "right" or shared_object) else "obj_left"
    fingertip_pattern = tuple(
      f"contact_{side}_{f}_tip"
      for f in ("thumb", "index", "middle", "ring", "pinky")
    )
    sensors.append(ContactSensorCfg(
      name=f"{p}_fingertip_contact",
      primary=ContactMatch(mode="site", pattern=fingertip_pattern, entity="hand"),
      secondary=ContactMatch(mode="body", pattern=obj_pattern, entity=obj_entity),
      fields=("found", "force"), reduce="netforce", secondary_policy="first",
    ))
    sensors.append(ContactSensorCfg(
      name=f"{p}_fingertip_penetration",
      primary=ContactMatch(mode="site", pattern=fingertip_pattern, entity="hand"),
      secondary=ContactMatch(mode="body", pattern=obj_pattern, entity=obj_entity),
      fields=("found", "dist"), reduce="mindist", secondary_policy="first",
    ))
  cfg.scene.sensors = tuple(sensors)

  tactile_obs = {
    "mano_joints_vel": ObservationTermCfg(
      func=mt_mdp.mano_joints_vel, params={"command_name": "motion"}),
    "mano_joints_vel_delta": ObservationTermCfg(
      func=mt_mdp.mano_joints_vel_delta, params={"command_name": "motion"}),
    "ref_contact_flags": ObservationTermCfg(
      func=mt_mdp.ref_contact_flags, params={"command_name": "motion"}),
  }
  for side in sides:
    p = side_pref[side]
    tactile_obs[f"{p}_contact_force"] = ObservationTermCfg(
      func=mt_mdp.contact_force,
      params={
        "sensor_name": f"{p}_fingertip_contact",
        "command_name": "motion",
        "side": side,
      },
    )
    tactile_obs[f"{p}_contact_force_history"] = ObservationTermCfg(
      func=mt_mdp.contact_force_history,
      params={
        "sensor_name": f"{p}_fingertip_contact",
        "command_name": "motion",
        "side": side,
        "history_len": 3,
      },
    )
  cfg.observations["actor"].terms.update(tactile_obs)
  cfg.observations["critic"].terms.update(tactile_obs)
  # Privileged-to-critic only: actor stays blind to object geometry.
  cfg.observations["critic"].terms["gt_tips_distance"] = ObservationTermCfg(
    func=mt_mdp.mano_tips_distance_obs, params={"command_name": "motion"})
  cfg.observations["critic"].terms["hand_obj_distance"] = ObservationTermCfg(
    func=mt_mdp.hand_obj_distance, params={"command_name": "motion"})

  cfg.terminations["dof_vel_sanity"] = TerminationTermCfg(
    func=mt_mdp.dof_vel_sanity,
    params={"max_dof_vel": 200.0, "asset_cfg": SceneEntityCfg("hand")},
  )

  object_obs = {
    "obj_pos_relative": ObservationTermCfg(
      func=mt_mdp.obj_pos_relative, params={"command_name": "motion"}),
    "obj_quat": ObservationTermCfg(
      func=mt_mdp.obj_quat, params={"command_name": "motion"}),
    "obj_vel": ObservationTermCfg(
      func=mt_mdp.obj_vel, params={"command_name": "motion"}),
    "obj_angvel": ObservationTermCfg(
      func=mt_mdp.obj_angvel, params={"command_name": "motion"}),
    "obj_pos_delta": ObservationTermCfg(
      func=mt_mdp.obj_pos_delta, params={"command_name": "motion"}),
    "obj_vel_delta": ObservationTermCfg(
      func=mt_mdp.obj_vel_delta, params={"command_name": "motion"}),
    "obj_quat_delta": ObservationTermCfg(
      func=mt_mdp.obj_quat_delta, params={"command_name": "motion"}),
    "obj_angvel_delta": ObservationTermCfg(
      func=mt_mdp.obj_angvel_delta, params={"command_name": "motion"}),
    "future_obj_pos_delta": ObservationTermCfg(
      func=mt_mdp.future_obj_pos_delta, params={"command_name": "motion"}),
    "future_obj_vel": ObservationTermCfg(
      func=mt_mdp.future_obj_vel, params={"command_name": "motion"}),
  }
  cfg.observations["actor"].terms.update(object_obs)
  cfg.observations["critic"].terms.update(object_obs)

  for side in sides:
    p = side_pref[side]
    for name, fn, w, scale in OBJECT_REWARD_TERMS:
      cfg.rewards[f"{p}_{name}"] = RewardTermCfg(
        func=fn, weight=w * args.object_reward_mult,
        params={"command_name": "motion", "side": side, "scale": scale},
      )

  cfg.terminations["obj_pos_diverged"] = TerminationTermCfg(
    func=mt_mdp.obj_pos_diverged,
    params={"command_name": "motion", "threshold": 0.06, "grace_steps": 15},
  )
  cfg.terminations["obj_rot_diverged"] = TerminationTermCfg(
    func=mt_mdp.obj_rot_diverged,
    params={"command_name": "motion", "threshold_deg": 90.0, "grace_steps": 15},
  )
  # cfg.terminations["contact_missed_too_long"] = TerminationTermCfg(
  #   func=mt_mdp.contact_missed_too_long,
  #   params={"command_name": "motion", "threshold_steps": args.contact_miss_t, "grace_steps": 15},
  # )
  if not getattr(args, "disable_velocity_sanity", False):
    cfg.terminations["joint_vel_sanity"] = TerminationTermCfg(
      func=mt_mdp.joint_vel_sanity,
      params={"max_joint_vel": 200.0, "asset_cfg": SceneEntityCfg("hand")},
    )
    cfg.terminations["obj_lin_vel_sanity"] = TerminationTermCfg(
      func=mt_mdp.obj_lin_vel_sanity,
      params={"command_name": "motion", "max_obj_vel": 100.0},
    )
    cfg.terminations["obj_ang_vel_sanity"] = TerminationTermCfg(
      func=mt_mdp.obj_ang_vel_sanity,
      params={"command_name": "motion", "max_obj_angvel": 200.0},
    )

  # ManipTrans-style curriculum: object termination tightening + gravity ramp.
  # curriculum_scale=0 disables. 1.0 = literal ManipTrans (gravity ep 80, term ep 133).
  if args.curriculum_scale > 0:
    from mjlab.envs.mdp.curriculums import termination_curriculum
    cfg.curriculum = {}
    if not args.disable_term_tightening:
      pos_stages, rot_stages = mt_mdp.build_obj_term_stages(args.curriculum_scale)
      cfg.curriculum["obj_pos_thr"] = CurriculumTermCfg(
        func=termination_curriculum,
        params={"termination_name": "obj_pos_diverged", "stages": pos_stages},
      )
      cfg.curriculum["obj_rot_thr"] = CurriculumTermCfg(
        func=termination_curriculum,
        params={"termination_name": "obj_rot_diverged", "stages": rot_stages},
      )
    cfg.curriculum["gravity"] = CurriculumTermCfg(
      func=mt_mdp.gravity_curriculum,
      params={
        "schedule_steps": (
          0 if args.gravity_constant else int(1920 * args.curriculum_scale)
        ),
        "full_g": args.gravity_full_g,
      },
    )
  return cfg


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--robot", required=True)
  p.add_argument("--side", required=True, choices=["right", "left", "bimanual"])
  p.add_argument("--input_dir", required=True,
    help="Pool root for object meshes (resolves <input_dir>/<pool_rel_dir>/<mesh_rel>).")
  p.add_argument("--motion_file", required=True,
    help="Packed motion .pt produced by package_motion_batch.py.")
  p.add_argument("--index", type=int, default=None,
    help="Optional: select one motion (by row index) inside the packed .pt for "
         "single-reference training. Default None = train on all M motions.")
  p.add_argument("--num_envs", type=int, required=True)
  p.add_argument("--max_iterations", type=int, default=1000000)
  p.add_argument("--save_interval", type=int, default=100)
  p.add_argument("--obs_clip", type=float, default=5.0)
  p.add_argument("--obj_density", type=float, default=800.0)
  p.add_argument("--contact_match_weight", type=float, default=1.0)
  p.add_argument("--contact_match_beta", type=float, default=40.0,
    help="Approach-shaping decay for contact_match: exp(-β·ref_dist). "
         "Lower β widens the basin of attraction (more gradient at long "
         "distance, less precision near contact). Default 40.0 (sharp).")
  p.add_argument("--object_reward_mult", type=float, default=1.0)
  p.add_argument("--contact_miss_t", type=int, default=999999,
    help="Per-finger consecutive-miss frame threshold for the "
         "contact_missed_too_long termination. Effectively disabled at the "
         "default; lower it once Metrics/motion/contact_miss_max_* histograms "
         "settle on what natural streaks look like.")
  p.add_argument("--base_checkpoints", nargs="+", required=True,
    help="1 path for --side {right,left}; 2 paths [right, left] for --side bimanual.")
  p.add_argument("--curriculum_scale", type=float, default=0.0,
    help="Multiplier on ManipTrans schedule_steps. 0 (default) = curriculum off. "
         "1.0 = literal ManipTrans (gravity ep 80, obj term ep 133). 10.0 = "
         "stretched (ep 800 / 1333). Tightens obj_pos/rot_diverged thresholds "
         "and ramps gravity 0->-9.81 m/s^2.")
  p.add_argument("--gravity_full_g", type=float, default=9.81,
    help="Final gravity magnitude in m/s^2 passed to gravity_curriculum. "
         "0 disables gravity (ramps from 0 -> 0).")
  p.add_argument("--gravity_constant", action="store_true",
    help="Skip the gravity ramp; gravity is held at -gravity_full_g from "
         "step 0 (sets schedule_steps=0 so gravity_curriculum returns frac=1).")
  p.add_argument("--disable_velocity_sanity", action="store_true",
    help="Skip the joint/obj_lin/obj_ang velocity_sanity terminations. nan_guard still active.")
  p.add_argument("--disable_term_tightening", action="store_true",
    help="Skip the obj_pos_thr/obj_rot_thr curriculum stages. Termination "
         "thresholds stay at the bare TerminationTermCfg defaults (6cm pos, "
         "90deg rot) throughout training.")
  p.add_argument("--pin_mode", choices=["none", "xfrc", "actuated", "hard"], default="none",
    help="Object pin / soft-attractor mode.")
  p.add_argument("--xfrc_omega_n", type=float, default=0.0,
    help="Natural frequency (rad/s) for mass-normalized xfrc PD attractor. "
         "Single knob: kp = m·ω², kv = 2ζ·sqrt(m·kp). Required when pin_mode=xfrc. "
         "Auto-detects per-object mass; logs ω_n + derived kp/kv to wandb.")
  p.add_argument("--xfrc_omega_n_end", type=float, default=None,
    help="If set, ω_n linearly decays from --xfrc_omega_n to this over "
         "--xfrc_omega_schedule_steps env-steps. Default None = no decay.")
  p.add_argument("--xfrc_omega_delay_steps", type=int, default=0,
    help="Hold ω at omega_n_start for this many env-steps before starting the linear decay. "
    "Note: 1 PPO iter = num_steps_per_env (32) env-steps; e.g. 200 iters = 6400 env-steps.")
  p.add_argument("--xfrc_omega_schedule_steps", type=int, default=0,
    help="Env-steps over which ω_n decays. 0 = held constant at --xfrc_omega_n.")
  p.add_argument("--xfrc_zeta", type=float, default=1.0,
    help="Damping ratio for the xfrc attractor. 1.0 = critical damping.")
  p.add_argument("--xfrc_kp_rot", type=float, default=0.0,
    help="Rotational P gain (raw scalar; ignored if --xfrc_omega_rot > 0).")
  p.add_argument("--xfrc_kv_rot", type=float, default=0.0,
    help="Rotational D gain (raw scalar; ignored if --xfrc_omega_rot > 0).")
  p.add_argument("--xfrc_omega_rot", type=float, default=0.0,
    help="If > 0, switches xfrc rotation PD to anisotropic inertia-tensor mode: "
    "τ = ω²·I_world·axis_angle + 2ζω·I_world·Δω. Overrides --xfrc_kp_rot/_kv_rot.")
  p.add_argument("--xfrc_zeta_rot", type=float, default=1.0,
    help="Damping ratio for anisotropic rotation PD (only if --xfrc_omega_rot > 0).")
  p.add_argument("--residual_action_scale", type=float, default=1.0)
  p.add_argument("--init_std", type=float, default=0.37)
  p.add_argument("--wandb_project", required=True)
  p.add_argument("--wandb_entity", required=True)
  p.add_argument("--wandb_tags", type=str, default="")
  p.add_argument("--group_name", required=True)
  p.add_argument("--exp_name", required=True)
  p.add_argument("--run_name", required=True)
  p.add_argument("--gpu", type=int, default=0)
  args = p.parse_args()

  if args.side == "bimanual" and len(args.base_checkpoints) != 2:
    raise ValueError(
      f"--side bimanual requires 2 --base_checkpoints; got {len(args.base_checkpoints)}.")
  if args.side in ("right", "left") and len(args.base_checkpoints) != 1:
    raise ValueError(
      f"--side {args.side} requires 1 --base_checkpoints; got {len(args.base_checkpoints)}.")

  configure_torch_backends()
  device = f"cuda:{args.gpu}"
  task_id = f"mjlab-maniptrans-{args.robot}-{args.side}"

  cfg = build_env_cfg(args)

  # NaN safety net: sanitize observations + terminate any env whose physics
  # state went NaN. Robust against transient finger penetration that produces
  # NaN dynamics; otherwise rsl_rl's check_nan kills the whole run.
  for group_name in ("actor", "critic"):
    if group_name in cfg.observations:
      cfg.observations[group_name].nan_policy = "sanitize"
  cfg.terminations["nan_guard"] = TerminationTermCfg(
    func=mt_mdp.nan_guard,
    params={"command_name": "motion", "asset_cfg": SceneEntityCfg("hand")},
  )

  # Optional object pin / xfrc soft-attractor.
  if args.pin_mode != "none":
    motion_cmd = cfg.commands["motion"]
    motion_cmd.pin_objects = True
    motion_cmd.pin_mode = args.pin_mode
    if args.pin_mode == "xfrc":
      if args.xfrc_omega_n <= 0:
        raise ValueError(f"--pin_mode=xfrc requires --xfrc_omega_n > 0; got {args.xfrc_omega_n}")
      motion_cmd.xfrc_kp_rot = float(args.xfrc_kp_rot)
      motion_cmd.xfrc_kv_rot = float(args.xfrc_kv_rot)
      motion_cmd.xfrc_omega_rot = float(args.xfrc_omega_rot)
      motion_cmd.xfrc_zeta_rot = float(args.xfrc_zeta_rot)
      # Mass-normalized PD: kp = m·ω², kv = 2ζ·sqrt(m·kp). Curriculum overwrites
      # cfg.xfrc_kp_pos/kv_pos each step from auto-detected obj mass + ω_n schedule;
      # logs ω_n + kp + kv to wandb (like gravity_z).
      motion_cmd.xfrc_kp_pos = 0.0  # placeholder; curriculum sets at step 0
      motion_cmd.xfrc_kv_pos = 0.0
      if cfg.curriculum is None:
        cfg.curriculum = {}
      omega_end = (args.xfrc_omega_n if args.xfrc_omega_n_end is None
                   else args.xfrc_omega_n_end)
      cfg.curriculum["xfrc_omega"] = CurriculumTermCfg(
        func=mt_mdp.xfrc_curriculum,
        params={
          "command_name": "motion",
          "omega_n_start": float(args.xfrc_omega_n),
          "omega_n_end": float(omega_end),
          "schedule_steps": int(args.xfrc_omega_schedule_steps),
          "delay_steps": int(args.xfrc_omega_delay_steps),
          "zeta": float(args.xfrc_zeta),
        },
      )
      rot_str = (f"ω_rot={args.xfrc_omega_rot} ζ_rot={args.xfrc_zeta_rot} (anisotropic tensor)"
                 if args.xfrc_omega_rot > 0
                 else f"kp_rot={args.xfrc_kp_rot} kv_rot={args.xfrc_kv_rot}")
      print(f"[train_residual] pin_mode=xfrc  ω_n={args.xfrc_omega_n}"
            f" → {omega_end} over {args.xfrc_omega_schedule_steps} steps; "
            f"ζ={args.xfrc_zeta}; {rot_str}",
            flush=True)

  base_dims = [_read_base_dims(p) for p in args.base_checkpoints]
  if len(base_dims) == 2 and base_dims[0] != base_dims[1]:
    raise ValueError(
      f"Per-side bases disagree on dims: right={base_dims[0]} vs left={base_dims[1]}.")
  base_obs_dim, base_action_dim = base_dims[0]

  env = ManagerBasedRlEnv(cfg, device=device)

  perside_term_specs: list[tuple[int, bool]] | None = None
  n_wrist_per_side = 0
  if args.side == "bimanual":
    active = env.observation_manager.active_terms["actor"]
    shapes = env.observation_manager.group_obs_term_dim["actor"]
    perside_term_specs = []
    accum = 0
    for name, shape in zip(active, shapes):
      bim_dim = int(shape[-1]) if isinstance(shape, (tuple, list)) else int(shape)
      is_global = name in GLOBAL_HAND_OBS_TERMS
      if not is_global and bim_dim % 2 != 0:
        raise RuntimeError(
          f"Per-side hand obs term {name!r} has odd bimanual dim {bim_dim}; "
          f"either it should be in GLOBAL_HAND_OBS_TERMS or bimanual layout is broken.")
      per_side_dim = bim_dim if is_global else bim_dim // 2
      perside_term_specs.append((per_side_dim, is_global))
      accum += per_side_dim
      if accum == base_obs_dim:
        break
      if accum > base_obs_dim:
        raise RuntimeError(
          f"Per-side hand obs accum {accum} overshot base_obs_dim {base_obs_dim} "
          f"at term {name!r}; check GLOBAL_HAND_OBS_TERMS membership.")
    if accum != base_obs_dim:
      raise RuntimeError(
        f"Per-side hand obs prefix sum {accum} != base_obs_dim {base_obs_dim}.")
    bimanual_n_wrist = env.action_manager.get_term("maniptrans")._n_wrist
    if bimanual_n_wrist % 2 != 0:
      raise RuntimeError(f"Bimanual n_wrist={bimanual_n_wrist} is odd.")
    n_wrist_per_side = bimanual_n_wrist // 2
    env_action_dim = env.action_manager.total_action_dim
    if env_action_dim != 2 * base_action_dim:
      raise RuntimeError(
        f"Bimanual env action_dim ({env_action_dim}) != 2 * base_action_dim "
        f"({2 * base_action_dim}).")

  agent_cfg = load_rl_cfg(task_id)
  agent_cfg.max_iterations = args.max_iterations
  agent_cfg.save_interval = args.save_interval
  agent_cfg.algorithm.learning_rate = 2.0e-4
  agent_cfg.algorithm.schedule = "fixed"
  agent_cfg.algorithm.value_loss_coef = 4.0
  agent_cfg.experiment_name = f"maniptrans_{args.robot}_stage2"
  agent_cfg.run_name = args.run_name
  agent_cfg.wandb_project = args.wandb_project
  agent_cfg.logger = "wandb"
  agent_cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": args.init_std,
    "std_type": "scalar",
  }

  train_cfg = asdict(agent_cfg)
  train_cfg["actor"]["class_name"] = (
    "mjlab.tasks.maniptrans.rl.residual_actor.ResidualActor"
  )
  train_cfg["actor"]["base_checkpoints"] = list(args.base_checkpoints)
  train_cfg["actor"]["base_obs_dim"] = base_obs_dim
  train_cfg["actor"]["base_action_dim"] = base_action_dim
  train_cfg["actor"]["residual_action_scale"] = args.residual_action_scale
  if perside_term_specs is not None:
    train_cfg["actor"]["perside_term_specs"] = perside_term_specs
    train_cfg["actor"]["n_wrist_per_side"] = n_wrist_per_side

  if args.obs_clip > 0:
    train_cfg["actor"]["obs_clip"] = args.obs_clip
    clipped_cls = "mjlab.tasks.maniptrans.rl.clipped_mlp_model.ClippedMLPModel"
    train_cfg["critic"]["class_name"] = clipped_cls
    train_cfg["critic"]["obs_clip"] = args.obs_clip

  log_dir = (
    Path("logs") / "rsl_rl" / train_cfg["experiment_name"]
    / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{train_cfg['run_name']}"
  )
  log_dir.mkdir(parents=True, exist_ok=True)

  env_wrapped = RslRlVecEnvWrapper(env)
  runner = MjlabOnPolicyRunner(env_wrapped, train_cfg, str(log_dir), device)

  if wandb.run is None:
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    wandb.init(
      project=args.wandb_project,
      entity=args.wandb_entity,
      name=train_cfg["run_name"],
      dir=str(log_dir),
      tags=tags if tags else None,
      config={"group_name": args.group_name, "exp_name": args.exp_name},
    )

  runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
  env.close()


if __name__ == "__main__":
  main()
