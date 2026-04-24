"""Train Stage 1 with pinned (kinematic) objects.

Base: objects present in the scene and pinned to the reference trajectory,
so the hand learns to track MANO poses while conforming to the object's
physical shape. Opt-in augmentations via CLI flags:

- `--enable_tactile`: tactile tier-1 obs + dof_vel_sanity + contact_missing
  terminations (see .task/15.md).
- `--enable_object_term`: object divergence / velocity terminations for
  pin_interval > 1 runs.
- `--enable_object_obs_{actor,critic}`: add ManipTrans Stage 2 object obs
  to actor / critic groups (independent, for asymmetric actor/critic setups).
- `--enable_object_rew` (+ `--object_reward_mult`): add ManipTrans Stage 2
  object tracking rewards. Lets a single policy learn hand + contact +
  object tracking in one shot instead of the ManipTrans two-stage split.
- `--actor_no_{hand_obj_distance,gt_tips_distance}`: strip object-geometry
  tactile obs from the actor group only (critic keeps them). Used for the
  blind-actor / privileged-critic ablation.
"""

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import wandb

import mjlab.tasks.maniptrans.config  # noqa: F401
from mjlab.asset_zoo.objects.entity import get_actuated_object_cfg, get_object_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.maniptrans import mdp as mt_mdp
from mjlab.tasks.maniptrans.mdp import ManipTransActionCfg, ManipTransCommandCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends


def build_env_cfg(args):
  """Build the full env cfg from an argparse namespace.

  Shared between `train_stage1_pinned.py` and `train_stage2.py` so Stage 2
  residual training reuses the exact same obs/reward/termination wiring as
  the Stage 1 base. Stage 2 callers call this, then apply Stage 2 overrides
  on the returned cfg (`motion_cmd.pin_objects = False`) and on the actor
  class (`train_cfg["actor"]["residual_action_scale"] = ...`, consumed by
  `rl/residual_actor.py::ResidualActor.__init__`).
  """
  task_id = f"mjlab-maniptrans-{args.robot}-{args.side}"
  cfg = load_env_cfg(task_id)
  cfg.scene.num_envs = args.num_envs

  # Physics config anchored on facebookresearch/spider (mjwp.py:78-107), the
  # mujoco-warp-native upstream our hand MJCFs derive from, plus MuJoCo docs'
  # elliptic-cone recommendation (modeling.rst:605) for fine grasping. Single
  # source of truth for both inspire and xhand — the per-side <option> blocks
  # inside the inspire MJCFs are redundant and should be removed.
  # o_solref / o_solimp intentionally left at MuJoCo defaults: spider's
  # d0=0.0 (zero impedance at contact) and width=0.03 are tuned for MPC-style
  # control and caused NaN observations in RL within ~13 iterations.
  cfg.sim.mujoco.timestep = float(os.environ.get("SIM_TIMESTEP", "0.01"))
  cfg.sim.mujoco.iterations = 20
  cfg.sim.mujoco.ls_iterations = 50
  cfg.sim.mujoco.integrator = "implicitfast"
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.mujoco.ccd_iterations = args.ccd_iterations
  # Ablation sweep hook: SIM_SOLIMP="d0,dmid,width,midpoint,power" overrides
  # the global contact solver impedance. Unset = MuJoCo default.
  _solimp_env = os.environ.get("SIM_SOLIMP", "").strip()
  if _solimp_env:
    cfg.sim.mujoco.o_solimp = tuple(float(x) for x in _solimp_env.split(","))
  if os.environ.get("SIM_DECIMATION"):
    cfg.decimation = int(os.environ["SIM_DECIMATION"])
  if os.environ.get("SIM_IMPRATIO"):
    cfg.sim.mujoco.impratio = float(os.environ["SIM_IMPRATIO"])

  # Set action params
  action_cfg = cfg.actions["maniptrans"]
  assert isinstance(action_cfg, ManipTransActionCfg)
  action_cfg.wrist_residual_scale = args.wrist_residual_scale
  action_cfg.finger_residual_scale = args.finger_residual_scale

  # Set command params
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, ManipTransCommandCfg)
  motion_cmd.motion_file = args.motion_file
  motion_cmd.pin_objects = True
  motion_cmd.pin_mode = args.pin_mode
  motion_cmd.xfrc_kp_pos = args.xfrc_kp_pos
  motion_cmd.xfrc_kv_pos = args.xfrc_kv_pos
  motion_cmd.xfrc_kp_rot = args.xfrc_kp_rot
  motion_cmd.xfrc_kv_rot = args.xfrc_kv_rot
  motion_cmd.pin_interval = args.pin_interval
  motion_cmd.adaptive_pin = args.adaptive_pin
  motion_cmd.pin_pos_threshold = args.pin_pos_threshold
  motion_cmd.pin_rot_threshold = args.pin_rot_threshold

  # --- No-object early branch (Stage 1 pure hand imitation) ---
  # Skips object entity, contact sensors, contact_match rewards, tactile obs,
  # object obs/rew/term. The policy trains on MANO reference alone (wrist +
  # fingertips + level1/level2 joint tracking, which all come from the motion
  # command, not the object). Tip tracks against the MANO tip in free space
  # rather than landing on a physical object surface.
  if getattr(args, "no_object", False):
    incompatible = []
    if args.enable_tactile:
      incompatible.append("--enable_tactile")
    if args.enable_object_obs_actor:
      incompatible.append("--enable_object_obs_actor")
    if args.enable_object_obs_critic:
      incompatible.append("--enable_object_obs_critic")
    if args.enable_object_rew:
      incompatible.append("--enable_object_rew")
    if args.enable_object_term:
      incompatible.append("--enable_object_term")
    if incompatible:
      raise ValueError(
        f"--no_object is incompatible with {incompatible}; all of these "
        "require the object entity to exist in the scene."
      )
    motion_cmd.pin_objects = False
    motion_cmd.object_entity_names = None
    # Drop contact_match rewards (added by config/base.py, keyed
    # r/l_{finger}_contact_match).
    for _key in list(cfg.rewards.keys()):
      if _key.endswith("_contact_match"):
        del cfg.rewards[_key]
    # No contact sensors (no object to contact against).
    cfg.scene.sensors = ()
    return cfg

  # Add object entities
  data_dir = Path(args.data_dir)
  with open(data_dir / "task_info.json") as f:
    task_info = json.load(f)
  right_obj_dir = str(data_dir / task_info["right_object_mesh_dir"])
  left_obj_dir = str(data_dir / task_info["left_object_mesh_dir"])
  # Per-object mesh scales from task_info.json (written by the dataset
  # preprocessing script, e.g. `src/preprocess/dataset/oakink2.py`). Default
  # 1.0 when the field is absent (back-compat with older preprocessed data).
  # Mass scales ~ scale^3 since density is unchanged.
  right_mesh_scale = float(task_info.get("right_object_mesh_scale", 1.0))
  left_mesh_scale = float(task_info.get("left_object_mesh_scale", 1.0))

  # Multi-trajectory runs share a single set of object entities across envs,
  # so every trajectory in the batch must refer to the same underlying
  # object. We compare the *relative* object mesh dirs from each traj's
  # task_info.json (typically `objects/O02@…@…`, the canonical object id)
  # and the per-object mesh scales. Absolute paths differ per traj because
  # each sub-sequence has its own preprocessed object copy, but the object
  # id and scale carry the logical identity. Reject ad-hoc index mixes with
  # a clear error rather than silently loading mismatched objects.
  if getattr(args, "_all_data_dirs", None) is not None and len(args._all_data_dirs) > 1:
    indices = getattr(args, "indices", [])
    base_right_rel = task_info["right_object_mesh_dir"]
    base_left_rel = task_info["left_object_mesh_dir"]
    mismatches: list[tuple[int, str, str, str, float, float]] = []
    for idx, other_dir in zip(indices[1:], args._all_data_dirs[1:]):
      with open(Path(other_dir) / "task_info.json") as f:
        other_info = json.load(f)
      other_right_rel = other_info["right_object_mesh_dir"]
      other_left_rel = other_info["left_object_mesh_dir"]
      other_right_scale = float(other_info.get("right_object_mesh_scale", 1.0))
      other_left_scale = float(other_info.get("left_object_mesh_scale", 1.0))
      if (other_right_rel, other_left_rel, other_right_scale, other_left_scale) != (
        base_right_rel, base_left_rel, right_mesh_scale, left_mesh_scale,
      ):
        mismatches.append((idx, other_right_rel, other_left_rel, other_right_scale, other_left_scale))  # type: ignore[arg-type]
    if mismatches:
      base = indices[0] if indices else "<unknown>"
      lines = [
        f"Multi-trajectory object mismatch: index {base} uses "
        f"right={base_right_rel!r} left={base_left_rel!r} "
        f"(scales {right_mesh_scale}/{left_mesh_scale}), but:",
      ]
      for m in mismatches:
        idx, r_rel, l_rel, r_s, l_s = m
        lines.append(
          f"  - index {idx}: right={r_rel!r} left={l_rel!r} (scales {r_s}/{l_s})"
        )
      lines.append(
        "All indices in a single run must share the same object ids and "
        "mesh scales. Use one of the curated groups (group1…group5) or a "
        "custom list with matching right/left object_mesh_dir and mesh_scale."
      )
      raise ValueError("\n".join(lines))

  # Shared-object detection. Some OakInk2 trajectories have both hands
  # manipulating the same physical item (hand-over, bimanual hold). Upstream
  # the preprocessing faithfully records this as `right_object_mesh_dir ==
  # left_object_mesh_dir` and `obj_right_* == obj_left_*` in motion.npz —
  # that's semantically correct. But if we then instantiate two separate
  # freejoint entities both pinned to the same reference trajectory, they
  # occupy the same voxel; between pin_interval snaps their independent xfrc
  # histories drift slightly and the rollout video shows two ghost objects
  # popping apart ("Harry Potter" duplication). Fix: detect the shared case
  # and build only one entity, then wire both sides of
  # `object_entity_names` to it. Per-side rewards/obs/tracking all
  # dereference via `commands.py`'s `object_entity_names[side]` loop, so
  # pointing both keys at "object_right" is sufficient — no reward or obs
  # logic changes needed. The `obj_list_rh[0] == obj_list_lh[0]` case also
  # implies the mesh scale is necessarily the same for both sides.
  shared_object = (right_obj_dir == left_obj_dir)

  if args.pin_mode == "actuated":
    cfg.scene.entities["object_right"] = get_actuated_object_cfg(
      right_obj_dir, "obj_right", density=args.obj_density,
      kp_pos=args.object_kp_pos, kv_pos=args.object_kv_pos,
      kp_rot=args.object_kp_rot, kv_rot=args.object_kv_rot,
      mesh_scale=right_mesh_scale,
    )
    if not shared_object:
      cfg.scene.entities["object_left"] = get_actuated_object_cfg(
        left_obj_dir, "obj_left", density=args.obj_density,
        kp_pos=args.object_kp_pos, kv_pos=args.object_kv_pos,
        kp_rot=args.object_kp_rot, kv_rot=args.object_kv_rot,
        mesh_scale=left_mesh_scale,
      )
  else:
    # "hard" and "xfrc" both use plain freejoint entities; they differ only
    # in how _update_command holds the object (root_state overwrite vs. xfrc PD).
    cfg.scene.entities["object_right"] = get_object_cfg(
      right_obj_dir, "obj_right", density=args.obj_density, mesh_scale=right_mesh_scale,
    )
    if not shared_object:
      cfg.scene.entities["object_left"] = get_object_cfg(
        left_obj_dir, "obj_left", density=args.obj_density, mesh_scale=left_mesh_scale,
      )

  # Set object entity names on command. In shared-object mode, both hands
  # reference the single `object_right` entity.
  if args.side == "right":
    if not shared_object:
      del cfg.scene.entities["object_left"]
    motion_cmd.object_entity_names = {"right": "object_right"}
  elif args.side == "left":
    if shared_object:
      motion_cmd.object_entity_names = {"left": "object_right"}
    else:
      del cfg.scene.entities["object_right"]
      motion_cmd.object_entity_names = {"left": "object_left"}
  else:
    if shared_object:
      motion_cmd.object_entity_names = {"right": "object_right", "left": "object_right"}
    else:
      motion_cmd.object_entity_names = {"right": "object_right", "left": "object_left"}

  # Multiply stratified per-finger contact_match weights by the CLI scalar
  # (global on/off + scale). Default 1.0 preserves config/base.py weights.
  for key, term in cfg.rewards.items():
    if key.endswith("_contact_match"):
      term.weight *= args.contact_match_weight
      term.params["beta"] = args.contact_match_beta
      term.params["gamma"] = args.contact_match_gamma
      term.params["tol"] = args.contact_match_tol
      term.params["force_cap"] = args.contact_match_force_cap

  # Apply pin-penalty weight (sign flipped; CLI arg is a positive magnitude).
  for key, term in cfg.rewards.items():
    if key.endswith("_pin_penalty"):
      term.weight = -args.pin_penalty_weight

  # Contact sensors (metrics only, no reward). In shared-object mode the
  # `object_left` entity doesn't exist; the left contact sensor points at
  # the single `object_right` entity (and its body name `obj_right`) so
  # both sides measure contacts against the same shared object.
  left_contact_entity = "object_right" if shared_object else "object_left"
  left_contact_pattern = "obj_right" if shared_object else "obj_left"
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="r_fingertip_contact",
      primary=ContactMatch(
        mode="site",
        pattern=("contact_right_thumb_tip", "contact_right_index_tip",
                 "contact_right_middle_tip", "contact_right_ring_tip",
                 "contact_right_pinky_tip"),
        entity="hand",
      ),
      secondary=ContactMatch(mode="body", pattern="obj_right", entity="object_right"),
      fields=("found", "force"),
      reduce="netforce",
      secondary_policy="first",
    ),
    ContactSensorCfg(
      name="l_fingertip_contact",
      primary=ContactMatch(
        mode="site",
        pattern=("contact_left_thumb_tip", "contact_left_index_tip",
                 "contact_left_middle_tip", "contact_left_ring_tip",
                 "contact_left_pinky_tip"),
        entity="hand",
      ),
      secondary=ContactMatch(mode="body", pattern=left_contact_pattern, entity=left_contact_entity),
      fields=("found", "force"),
      reduce="netforce",
      secondary_policy="first",
    ),
    # Penetration sensors — mindist reduction emits signed `dist` for the
    # deepest contact per fingertip site. Inspection-only; no reward/obs/term
    # consumer. See .progress/6-hand-cleanup/output/field_and_config_analysis.md.
    ContactSensorCfg(
      name="r_fingertip_penetration",
      primary=ContactMatch(
        mode="site",
        pattern=("contact_right_thumb_tip", "contact_right_index_tip",
                 "contact_right_middle_tip", "contact_right_ring_tip",
                 "contact_right_pinky_tip"),
        entity="hand",
      ),
      secondary=ContactMatch(mode="body", pattern="obj_right", entity="object_right"),
      fields=("found", "dist"),
      reduce="mindist",
      secondary_policy="first",
    ),
    ContactSensorCfg(
      name="l_fingertip_penetration",
      primary=ContactMatch(
        mode="site",
        pattern=("contact_left_thumb_tip", "contact_left_index_tip",
                 "contact_left_middle_tip", "contact_left_ring_tip",
                 "contact_left_pinky_tip"),
        entity="hand",
      ),
      secondary=ContactMatch(mode="body", pattern=left_contact_pattern, entity=left_contact_entity),
      fields=("found", "dist"),
      reduce="mindist",
      secondary_policy="first",
    ),
  )

  # --- Tactile tier 1 (gated by --enable_tactile) ----------------------------
  if args.enable_tactile:
    tactile_obs_core = {
      "mano_joints_vel": ObservationTermCfg(
        func=mt_mdp.mano_joints_vel,
        params={"command_name": "motion"},
      ),
      "mano_joints_vel_delta": ObservationTermCfg(
        func=mt_mdp.mano_joints_vel_delta,
        params={"command_name": "motion"},
      ),
      "r_contact_force": ObservationTermCfg(
        func=mt_mdp.contact_force,
        params={"sensor_name": "r_fingertip_contact"},
      ),
      "l_contact_force": ObservationTermCfg(
        func=mt_mdp.contact_force,
        params={"sensor_name": "l_fingertip_contact"},
      ),
      "r_contact_force_history": ObservationTermCfg(
        func=mt_mdp.contact_force_history,
        params={"sensor_name": "r_fingertip_contact", "history_len": 3},
      ),
      "l_contact_force_history": ObservationTermCfg(
        func=mt_mdp.contact_force_history,
        params={"sensor_name": "l_fingertip_contact", "history_len": 3},
      ),
      "ref_contact_flags": ObservationTermCfg(
        func=mt_mdp.ref_contact_flags,
        params={"command_name": "motion"},
      ),
    }
    gt_tips_distance_term = ObservationTermCfg(
      func=mt_mdp.mano_tips_distance_obs,
      params={"command_name": "motion"},
    )
    hand_obj_distance_term = ObservationTermCfg(
      func=mt_mdp.hand_obj_distance,
      params={"command_name": "motion"},
    )
    cfg.observations["actor"].terms.update(tactile_obs_core)
    cfg.observations["critic"].terms.update(tactile_obs_core)
    cfg.observations["critic"].terms["gt_tips_distance"] = gt_tips_distance_term
    if not args.actor_no_gt_tips_distance:
      cfg.observations["actor"].terms["gt_tips_distance"] = gt_tips_distance_term
    cfg.observations["critic"].terms["hand_obj_distance"] = hand_obj_distance_term
    if not args.actor_no_hand_obj_distance:
      cfg.observations["actor"].terms["hand_obj_distance"] = hand_obj_distance_term

    cfg.terminations["dof_vel_sanity"] = TerminationTermCfg(
      func=mt_mdp.dof_vel_sanity,
      params={
        "max_dof_vel": 200.0,
        "asset_cfg": SceneEntityCfg("hand"),
      },
    )
    if not args.no_contact_missing:
      cfg.terminations["r_contact_missing"] = TerminationTermCfg(
        func=mt_mdp.contact_expected_but_missing,
        params={
          "command_name": "motion",
          "force_history_key": "_contact_force_history_r_fingertip_contact",
          "side": "right",
          "dist_threshold": 0.005,
          "grace_steps": 15,
        },
      )
      cfg.terminations["l_contact_missing"] = TerminationTermCfg(
        func=mt_mdp.contact_expected_but_missing,
        params={
          "command_name": "motion",
          "force_history_key": "_contact_force_history_l_fingertip_contact",
          "side": "left",
          "dist_threshold": 0.005,
          "grace_steps": 15,
        },
      )

  # --- Object obs + rewards (gated by --enable_object_obs_{actor,critic}
  #     and --enable_object_rew) ---
  if args.enable_object_obs_actor or args.enable_object_obs_critic:
    object_obs = {
      "obj_pos_relative": ObservationTermCfg(
        func=mt_mdp.obj_pos_relative,
        params={"command_name": "motion"},
      ),
      "obj_quat": ObservationTermCfg(
        func=mt_mdp.obj_quat,
        params={"command_name": "motion"},
      ),
      "obj_vel": ObservationTermCfg(
        func=mt_mdp.obj_vel,
        params={"command_name": "motion"},
      ),
      "obj_angvel": ObservationTermCfg(
        func=mt_mdp.obj_angvel,
        params={"command_name": "motion"},
      ),
      "obj_pos_delta": ObservationTermCfg(
        func=mt_mdp.obj_pos_delta,
        params={"command_name": "motion"},
      ),
      "obj_vel_delta": ObservationTermCfg(
        func=mt_mdp.obj_vel_delta,
        params={"command_name": "motion"},
      ),
      "obj_quat_delta": ObservationTermCfg(
        func=mt_mdp.obj_quat_delta,
        params={"command_name": "motion"},
      ),
      "obj_angvel_delta": ObservationTermCfg(
        func=mt_mdp.obj_angvel_delta,
        params={"command_name": "motion"},
      ),
      "future_obj_pos_delta": ObservationTermCfg(
        func=mt_mdp.future_obj_pos_delta,
        params={"command_name": "motion"},
      ),
      "future_obj_vel": ObservationTermCfg(
        func=mt_mdp.future_obj_vel,
        params={"command_name": "motion"},
      ),
    }
    if args.enable_object_obs_actor:
      cfg.observations["actor"].terms.update(object_obs)
    if args.enable_object_obs_critic:
      cfg.observations["critic"].terms.update(object_obs)

  if args.enable_object_rew:
    mult = args.object_reward_mult
    side_prefixes = {"right": "r", "left": "l"}
    for side in motion_cmd.sides:
      pref = side_prefixes[side]
      cfg.rewards[f"{pref}_obj_pos"] = RewardTermCfg(
        func=mt_mdp.obj_pos_error_exp,
        weight=5.0 * mult,
        params={"command_name": "motion", "side": side, "scale": 80.0},
      )
      cfg.rewards[f"{pref}_obj_rot"] = RewardTermCfg(
        func=mt_mdp.obj_rot_error_exp,
        weight=1.0 * mult,
        params={"command_name": "motion", "side": side, "scale": 3.0},
      )
      cfg.rewards[f"{pref}_obj_vel"] = RewardTermCfg(
        func=mt_mdp.obj_vel_error_exp,
        weight=0.1 * mult,
        params={"command_name": "motion", "side": side, "scale": 1.0},
      )
      cfg.rewards[f"{pref}_obj_angvel"] = RewardTermCfg(
        func=mt_mdp.obj_angvel_error_exp,
        weight=0.1 * mult,
        params={"command_name": "motion", "side": side, "scale": 1.0},
      )

  # Object termination criteria (Stage 2 defaults) — active when --enable_object_term
  # is set. Kills the episode when the sim object drifts/spins/flies away from the
  # reference, preventing poisoned rollouts under pin_interval > 1 (temporal curriculum).
  if args.enable_object_term:
    cfg.terminations["obj_pos_diverged"] = TerminationTermCfg(
      func=mt_mdp.obj_pos_diverged,
      params={"command_name": "motion", "threshold": 0.15, "grace_steps": 15},
    )
    cfg.terminations["obj_rot_diverged"] = TerminationTermCfg(
      func=mt_mdp.obj_rot_diverged,
      params={"command_name": "motion", "threshold_deg": 90.0, "grace_steps": 15},
    )
    cfg.terminations["velocity_sanity"] = TerminationTermCfg(
      func=mt_mdp.velocity_sanity,
      params={
        "command_name": "motion",
        "max_obj_vel": 100.0,
        "max_obj_angvel": 200.0,
        "max_joint_vel": 200.0,
        "asset_cfg": SceneEntityCfg("hand"),
      },
    )

  return cfg


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--robot", required=True)
  p.add_argument("--side", required=True)
  p.add_argument("--index_path", required=True,
    help="Path to the dataset index CSV (e.g. /home/nas_main/.cache/datasets/oakink2/index.csv).")
  p.add_argument("--indices", type=int, nargs="+", required=True,
    help="One or more integer indices into the index CSV. Single = single-ref, "
         "multiple = multi-trajectory training (all must share the same objects).")
  p.add_argument("--output_dir", required=True,
    help="Preprocessing output directory (e.g. /home/nas_main/.cache/datasets/handphuma). "
         "Motion files at $OUTPUT_DIR/$ROBOT/$DATASET/$SCENE/$SUBSEQ/motion.npz, "
         "object data at $OUTPUT_DIR/$DATASET/$SCENE/$SUBSEQ.")
  p.add_argument("--obj_density", type=float, required=True)
  p.add_argument("--wrist_residual_scale", type=float, required=True)
  p.add_argument("--finger_residual_scale", type=float, required=True)
  p.add_argument("--contact_match_weight", type=float, required=True,
    help="Global multiplier on stratified per-finger contact_match weights.")
  p.add_argument("--contact_match_beta", type=float, required=True,
    help="Approach-shaping decay: exp(-beta · ref_dist). Default 40 /m.")
  p.add_argument("--contact_match_gamma", type=float, default=200.0,
    help="Depth-penalty decay for the contact bonus: exp(-gamma · max(-dist-tol, 0)). "
         "Default 200 /m = 0.2 per mm beyond tolerance.")
  p.add_argument("--contact_match_tol", type=float, default=0.002,
    help="Penetration tolerance (m): overlap below tol gets full bonus plateau. "
         "Default 0.002 (2 mm) — ManipTrans 'slight penetration permitted'.")
  p.add_argument("--contact_match_force_cap", type=float, default=30.0,
    help="Force magnitude cap (N) on the contact bonus: if fingertip ||F|| >= cap, "
         "the exp(-γ · depth) bonus is multiplied by 0 (zeroed). Approach-shaping "
         "term is unaffected. Default 30.0 (ProtoMotions threshold). Set to a very "
         "large value (e.g. 1e6) to effectively disable the cap.")
  p.add_argument("--pin_mode", choices=("hard", "actuated", "xfrc"), default="hard")
  p.add_argument("--pin_interval", type=int, default=6,
    help="For pin_mode=hard: fixed temporal pin interval T. T=1 pins every physics step "
         "(full hard-pin). T=N lets the object drift N-1 steps per cycle. Default 6.")
  p.add_argument("--adaptive_pin", action="store_true",
    help="Replace fixed-interval pinning with deviation-gated (per-side) adaptive "
         "pinning. Pin fires on a side when pos_dev>pin_pos_threshold OR "
         "rot_dev>pin_rot_threshold. Only wired for pin_mode=hard.")
  p.add_argument("--pin_pos_threshold", type=float, default=0.030,
    help="Adaptive pinning: position deviation threshold (m). Default 0.030 (3 cm).")
  p.add_argument("--pin_rot_threshold", type=float, default=1.5708,
    help="Adaptive pinning: rotation deviation threshold (rad). Default 1.5708 (90°).")
  p.add_argument("--pin_penalty_weight", type=float, default=0.0,
    help="Positive magnitude of the per-side pin-penalty weight. Applied as "
         "-pin_penalty_weight on {r,l}_pin_penalty reward terms. 0 disables. "
         "Under adaptive pinning the policy learns to avoid firing the pin.")
  # Gains for pin_mode="actuated" (6-joint position actuators):
  p.add_argument("--object_kp_pos", type=float, default=0.0)
  p.add_argument("--object_kv_pos", type=float, default=0.0)
  p.add_argument("--object_kp_rot", type=float, default=0.0)
  p.add_argument("--object_kv_rot", type=float, default=0.0)
  # Gains for pin_mode="xfrc" (DexMachina-style world-frame PD on freejoint):
  p.add_argument("--xfrc_kp_pos", type=float, default=0.0)
  p.add_argument("--xfrc_kv_pos", type=float, default=0.0)
  p.add_argument("--xfrc_kp_rot", type=float, default=0.0)
  p.add_argument("--xfrc_kv_rot", type=float, default=0.0)
  p.add_argument("--enable_tactile", action="store_true",
    help="Add tactile tier-1 obs (contact_force, contact_force_history, ref_contact_flags, "
         "gt_tips_distance, hand_obj_distance, mano_joints_vel{,_delta}) and terminations "
         "(dof_vel_sanity, r/l_contact_missing). See .task/15.md.")
  p.add_argument("--no_contact_missing", action="store_true",
    help="When --enable_tactile is set, skip the r/l_contact_missing terminations "
         "(keep the obs and dof_vel_sanity). Ablation: isolates the effect of the "
         "contact-missing kill signal from the tactile obs signal.")
  p.add_argument("--enable_object_term", action="store_true",
    help="Add Stage 2 object termination criteria: obj_pos_diverged (0.15 m), "
         "obj_rot_diverged (90 deg), velocity_sanity (obj_vel > 100, obj_angvel > 200, "
         "joint_vel > 200). Useful when training with pin_interval > 1 so the episode "
         "dies when the object drifts/spins/flies away instead of poisoning rollouts.")
  p.add_argument("--enable_object_obs_actor", action="store_true",
    help="Add the 10 ManipTrans Stage 2 object observations to the *actor* group: "
         "obj_{pos_relative,quat,vel,angvel} + their deltas + future_obj_{pos_delta,vel} "
         "(1-step lookahead). Use with --enable_object_obs_critic for the symmetric "
         "HCO setup; omit (while keeping --enable_object_obs_critic on) for the "
         "blind-actor / privileged-critic ablation.")
  p.add_argument("--enable_object_obs_critic", action="store_true",
    help="Add the 10 ManipTrans Stage 2 object observations to the *critic* group. "
         "Independent of --enable_object_obs_actor so asymmetric actor/critic obs "
         "spaces are possible. For the blind-actor ablation, set only this flag.")
  p.add_argument("--enable_object_rew", action="store_true",
    help="Add the 4 ManipTrans Stage 2 per-side object tracking rewards "
         "(obj_{pos,rot,vel,angvel}_error_exp) with ManipTrans weights "
         "(5.0/1.0/0.1/0.1, scales 80/3/1/1) scaled by --object_reward_mult. "
         "Does NOT add the ManipTrans contact_force_reward — the --contact_match_* "
         "path stays in charge of contact shaping. Use with pin_interval > 1 (and "
         "ideally --enable_object_term) so the policy actually has to hold the "
         "object between pin snaps rather than free-riding on the pin.")
  p.add_argument("--object_reward_mult", type=float, default=1.0,
    help="Multiplier on the object tracking reward weights. Only relevant with "
         "--enable_object_rew. 1.0 = ManipTrans Stage 2 weights exactly (obj_pos=5.0, "
         "obj_rot=1.0, obj_vel=0.1, obj_angvel=0.1). 0.0 = object obs added but zero "
         "reward (useful for ablating obs vs rew contribution). Fractional values "
         "sweep the weight axis.")
  p.add_argument("--actor_no_hand_obj_distance", action="store_true",
    help="When --enable_tactile is set, remove `hand_obj_distance` from the *actor* "
         "observation group only; the critic still sees it. Used for the blind-actor "
         "ablation to test whether object tracking reward alone can shape the policy "
         "without explicit object-position leakage in the actor inputs.")
  p.add_argument("--actor_no_gt_tips_distance", action="store_true",
    help="When --enable_tactile is set, remove `gt_tips_distance` from the *actor* "
         "observation group only; the critic still sees it. Parallel to "
         "--actor_no_hand_obj_distance; both together strip the two object-geometry-"
         "carrying obs from the actor for the strict blind-actor ablation.")
  p.add_argument("--no_object", action="store_true",
    help="Pure hand-imitation Stage 1: do NOT add object entities to the scene, "
         "do NOT install contact sensors, do NOT apply contact_match rewards. "
         "Motion still provides the MANO reference for wrist/fingertip tracking; "
         "the robot hand moves through empty air. Incompatible with --enable_tactile, "
         "--enable_object_obs_*, --enable_object_rew, --enable_object_term "
         "(all of which require the object to exist).")
  p.add_argument("--wandb_tags", type=str, default="",
    help="Comma-separated list of wandb tags for the run (see RUN-EXPERIMENT.md "
         "for the controlled vocabulary: single-reference, ppo, mujoco-warp, xhand, "
         "oakink2, hand-contact-object, etc.). Empty string = no tags.")
  p.add_argument("--ccd_iterations", type=int, default=50,
    help="MuJoCo CCD (continuous collision detection) iteration cap. MuJoCo "
         "default is 50. Bump to 100/150/200 to silence 'opt.ccd_iterations "
         "needs to be increased' warnings on dex-manip tasks with complex "
         "object meshes / fast finger-object impacts. Cost is usually <5%% "
         "wall-clock per step because only unconverged contact pairs hit "
         "the higher cap.")
  p.add_argument("--obs_clip", type=float, default=0.0,
    help="Observation clip value for the actor+critic MLPs. When > 0, both "
         "actor and critic classes are swapped to `ClippedMLPModel`, which "
         "applies `clamp(normalize(obs), -obs_clip, obs_clip)` BEFORE the MLP. "
         "This makes the policy robust to out-of-distribution observations, "
         "which is essential when the checkpoint will later be used as a "
         "frozen base for Stage 2 residual training (rsl_rl's default MLPModel "
         "has no clipping, so rare OOD values at inference time extrapolate to "
         "absurd actions and blow up the sim). Typical values: 5.0 (rl_games, "
         "sb3 defaults), 10.0 (more permissive). 0.0 = no clipping (default, "
         "backward compat with existing Stage 1 runs).")
  p.add_argument("--num_envs", type=int, required=True)
  p.add_argument("--max_iterations", type=int, required=True)
  p.add_argument("--save_interval", type=int, required=True)
  p.add_argument("--eval_interval", type=int, default=500,
    help="Run eval rollouts and log E_fingertip / E_wrist_pos / E_wrist_rot to "
         "wandb every this many training iterations. Also runs once at the "
         "final iteration regardless. Set to 0 to disable.")
  p.add_argument("--eval_rollouts_per_traj", type=int, default=20,
    help="How many rollouts per trajectory during periodic eval. Total eval env "
         "count = eval_rollouts_per_traj * len(indices). Trajectory assignment "
         "across eval envs is random uniform via the command's _resample_command, "
         "giving ~equal coverage of each traj in expectation.")
  p.add_argument("--wandb_project", required=True)
  p.add_argument("--wandb_entity", required=True,
    help="wandb entity (team/user). No default — set explicitly per run.")
  p.add_argument("--group_name", required=True,
    help="High-level ablation study this run belongs to (maps to wandb `group`). "
         "Groups all runs of a single study together in the wandb UI.")
  p.add_argument("--exp_name", required=True,
    help="Specific ablation variant within the group (maps to wandb `job_type`). "
         "Identifies which condition this run represents inside the group.")
  p.add_argument("--run_name", required=True)
  p.add_argument("--gpu", type=int, required=True)
  args = p.parse_args()

  # Resolve indices → motion_file and data_dir
  import csv
  with open(args.index_path) as f:
    rows = list(csv.DictReader(f))
  motion_files = []
  data_dirs = []
  for idx in args.indices:
    row = rows[idx]
    rel = row["dataset"] + "/" + row["filename"]
    motion_files.append(f"{args.output_dir}/{args.robot}/{rel}/motion.npz")
    data_dirs.append(f"{args.output_dir}/{row['dataset']}/{row['filename']}")
  first_rel = rows[args.indices[0]]["dataset"] + "/" + rows[args.indices[0]]["filename"]
  args.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]
  args.data_dir = data_dirs[0]
  args._all_data_dirs = data_dirs  # consumed by build_env_cfg for multi-traj object validation

  configure_torch_backends()
  device = f"cuda:{args.gpu}"
  task_id = f"mjlab-maniptrans-{args.robot}-{args.side}"
  cfg = build_env_cfg(args)

  # Create env
  env = ManagerBasedRlEnv(cfg, device=device)

  # --- Eval env (separate instance, smaller num_envs, play-mode sampling, no
  # imitation-quality terminations). Rebuilt via build_env_cfg with overridden
  # args.num_envs so it inherits all other training overrides (physics, obs,
  # rewards, object entities) exactly. ---
  eval_env = None
  eval_wrapped = None
  if args.eval_interval > 0:
    n_eval = args.eval_rollouts_per_traj * len(args.indices)
    _orig_num_envs = args.num_envs
    args.num_envs = n_eval
    try:
      eval_cfg = build_env_cfg(args)
    finally:
      args.num_envs = _orig_num_envs

    eval_motion_cmd = eval_cfg.commands["motion"]
    assert isinstance(eval_motion_cmd, ManipTransCommandCfg)
    eval_motion_cmd.sampling_mode = "start"

    # Strip imitation-quality terminations so each env runs the full motion.
    # Keep time_out + velocity_diverged as sim-blowup guards.
    for _term_name in ("fingertip_diverged", "dof_vel_sanity",
                       "r_contact_missing", "l_contact_missing"):
      eval_cfg.terminations.pop(_term_name, None)

    # Obs corruption off (only meant for training-time robustness).
    for _group_name in ("actor", "critic"):
      if _group_name in eval_cfg.observations:
        eval_cfg.observations[_group_name].enable_corruption = False

    eval_env = ManagerBasedRlEnv(eval_cfg, device=device)
    eval_wrapped = RslRlVecEnvWrapper(eval_env)

  # RL config
  agent_cfg = load_rl_cfg(task_id)
  agent_cfg.max_iterations = args.max_iterations
  agent_cfg.save_interval = args.save_interval
  agent_cfg.experiment_name = f"maniptrans_{args.robot}"
  agent_cfg.run_name = args.run_name
  agent_cfg.wandb_project = args.wandb_project
  agent_cfg.logger = "wandb"

  train_cfg = asdict(agent_cfg)

  # If --obs_clip > 0, swap both actor and critic to ClippedMLPModel so the
  # trained base is robust to OOD obs when later used as a frozen base for
  # Stage 2 residual training. See `rl/clipped_mlp_model.py` for rationale.
  if args.obs_clip > 0:
    clipped_cls = "mjlab.tasks.maniptrans.rl.clipped_mlp_model.ClippedMLPModel"
    train_cfg["actor"]["class_name"] = clipped_cls
    train_cfg["actor"]["obs_clip"] = args.obs_clip
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

  # --- Periodic eval hook via logger monkey-patch ---
  # OnPolicyRunner.learn() calls self.logger.log(it=..., ...) once per training
  # iteration. Wrap it to trigger evaluate_stage1 + wandb log at every
  # eval_interval (post-iteration) and at the final iteration. Keeps the
  # single learn() call intact (avoids per-chunk final-save + stop-writer
  # side effects of splitting learn() into multiple calls).
  if eval_wrapped is not None and args.eval_interval > 0:
    from mjlab.tasks.maniptrans.scripts.stage1_scorer import (
      evaluate_stage1, log_eval_to_wandb,
    )

    _orig_log = runner.logger.log
    _max_it = args.max_iterations
    _eval_iv = args.eval_interval

    def _log_with_eval(**kwargs):
      _orig_log(**kwargs)
      it = kwargs.get("it")
      if it is None:
        return
      # it is 0-indexed iteration just completed; fire eval at end of every
      # eval_interval completed iters and at the final iter.
      completed = it + 1
      fire = (completed % _eval_iv == 0) or (completed == _max_it)
      if not fire:
        return
      policy = runner.get_inference_policy(device=device)
      metrics = evaluate_stage1(eval_env, eval_wrapped, policy, device)
      log_eval_to_wandb(metrics, iter_idx=completed)
      g = metrics["global"]
      print(
        f"[eval @ iter {completed}/{_max_it}]  "
        f"E_fingertip={g['tip_pos_cm']:6.2f} cm  "
        f"E_wrist_pos={g['wrist_pos_cm']:6.2f} cm  "
        f"E_wrist_rot={g['wrist_rot_deg']:6.2f} deg  "
        f"(n_traj={g['n_trajectories_evaluated']}, n_envs={g['n_envs']})"
      )

    runner.logger.log = _log_with_eval

  runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
  env.close()
  if eval_env is not None:
    eval_env.close()


if __name__ == "__main__":
  main()
