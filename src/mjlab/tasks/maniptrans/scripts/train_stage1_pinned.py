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

  cfg.sim.mujoco.ccd_iterations = args.ccd_iterations

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

  # Hand friction
  hand_cfg = cfg.scene.entities["hand"]
  for col_cfg in hand_cfg.collisions:
    col_cfg.friction = (4.0, 0.01, 0.01)

  # Override contact match reward weight, beta, A, and eps from CLI
  for key, term in cfg.rewards.items():
    if key.endswith("_contact_match"):
      term.weight = args.contact_match_weight
      term.params["beta"] = args.contact_match_beta
      term.params["A"] = args.contact_match_A
      term.params["eps"] = args.contact_match_eps

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
  p.add_argument("--contact_match_weight", type=float, required=True)
  p.add_argument("--contact_match_beta", type=float, required=True)
  p.add_argument("--contact_match_A", type=float, required=True)
  p.add_argument("--contact_match_eps", type=float, default=0.05,
    help="Force threshold (N) for 'in contact' detection in contact_match reward. "
         "Below this, approach shaping exp(-beta*dist) applies; above, flat A. "
         "Default 0.05 matches v15. Lower values (e.g., 0.005) catch light-contact "
         "events on low-mass objects (e.g., 5.7g alcohol lamp lid on the right side).")
  p.add_argument("--pin_mode", choices=("hard", "actuated", "xfrc"), default="hard")
  p.add_argument("--pin_interval", type=int, default=6,
    help="For pin_mode=hard: fixed temporal pin interval T. T=1 pins every physics step "
         "(full hard-pin). T=N lets the object drift N-1 steps per cycle. Default 6.")
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
  p.add_argument("--wandb_tags", type=str, default="",
    help="Comma-separated list of wandb tags for the run (see RUN-EXPERIMENT.md "
         "for the controlled vocabulary: single-reference, ppo, mujoco-warp, xhand, "
         "oakink2, hand-contact-object, etc.). Empty string = no tags.")
  p.add_argument("--ccd_iterations", type=int, default=50,
    help="MuJoCo CCD (continuous collision detection) iteration cap. MuJoCo "
         "default is 50. Bump to 100/150/200 to silence 'opt.ccd_iterations "
         "needs to be increased' warnings on dex-manip tasks with complex "
         "object meshes / fast finger-object impacts. Cost is usually <5% "
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
  p.add_argument("--wandb_project", required=True)
  p.add_argument("--wandb_entity", required=True,
    help="wandb entity (team/user). No default — set explicitly per run.")
  p.add_argument("--run_name", required=True)
  p.add_argument("--gpu", type=int, required=True)
  args = p.parse_args()

  # Resolve indices → motion_file and data_dir
  import csv
  with open(args.index_path) as f:
    rows = list(csv.DictReader(f))
  motion_files = []
  for idx in args.indices:
    row = rows[idx]
    rel = row["dataset"] + "/" + row["filename"]
    motion_files.append(f"{args.output_dir}/{args.robot}/{rel}/motion.npz")
  first_rel = rows[args.indices[0]]["dataset"] + "/" + rows[args.indices[0]]["filename"]
  args.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]
  args.data_dir = f"{args.output_dir}/{rows[args.indices[0]]['dataset']}/{rows[args.indices[0]]['filename']}"

  configure_torch_backends()
  device = f"cuda:{args.gpu}"
  task_id = f"mjlab-maniptrans-{args.robot}-{args.side}"
  cfg = build_env_cfg(args)

  # Create env
  env = ManagerBasedRlEnv(cfg, device=device)

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
    )

  runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
  env.close()


if __name__ == "__main__":
  main()
