"""Per-motion scorer for ManipTrans Stage-2 residual policies (single-side).

Pins env i to motion i in the packed motion file (M motions => M envs), runs
each env for `--eval_seconds` of simulated time (short motions auto-cycle via
the motion command's wrap-around), and emits per-motion time-averaged metrics
+ per-motion SR@k pass/fail to a CSV row.

Adapted from `score_base_per_motion.py` (per-motion pinning + CSV writer) +
`score_perside.py` (ResidualActor construction, threshold-based SR@k, contact
telemetry). Right/left single-side only; bimanual not supported.

Output CSV columns (per motion):
  index, num_frame
  + 6 errors:  error_wrist_pos_r (cm), error_wrist_rot_r (deg),
               error_tip_pos_r (cm, mean of 5 fingers),
               error_obj_pos_r (cm), error_obj_rot_r (deg),
               error_joint_r (cm, mean of L1+L2)
  + 5 pass:    pass@0.5, pass@1.0, pass@1.5, pass@2.0, pass@3.0  (0/1)
  + 5*4=20 contact: contact_{ref_dist,pen,force,found}_{r|l}_{thumb,index,middle,ring,pinky}
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch

import mjlab.tasks.maniptrans.config  # noqa: F401  registers tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.maniptrans.mdp import ManipTransCommandCfg
from mjlab.tasks.maniptrans.mdp.commands import ManipTransCommand
from mjlab.tasks.maniptrans.scripts.train_residual import (
  _read_base_dims,
  build_env_cfg,
)
from mjlab.tasks.registry import load_rl_cfg
from mjlab.utils.torch import configure_torch_backends


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
SIDE_PREFIX = {"right": "r", "left": "l"}

TH_OBJ_POS = 0.03
TH_TIP_POS = 0.06
TH_OBJ_ROT = 30.0  # deg
TH_JOINT_POS = 0.08


def _rotmat_axis_angle_deg(R_diff: torch.Tensor) -> torch.Tensor:
  trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
  cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
  return torch.acos(cos_theta) * (180.0 / math.pi)


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return torch.stack([
    torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
    torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1),
    torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1),
  ], dim=-2)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--robot", required=True)
  p.add_argument("--side", default="right", choices=["right", "left"],
    help="Single-side only; bimanual not supported here.")
  p.add_argument("--motion_file", required=True)
  p.add_argument("--input_dir", required=True,
    help="Pool root for object meshes; resolves <input_dir>/<pool_rel_dir>/<mesh_rel>.")
  p.add_argument("--base_checkpoints", nargs="+", required=True,
    help="Single-hand: 1 base ckpt path.")
  p.add_argument("--residual_checkpoint", required=True)
  p.add_argument("--residual_action_scale", type=float, default=2.0)
  p.add_argument("--residual_init_std", type=float, default=0.37)
  p.add_argument("--obj_density", type=float, default=800.0)
  p.add_argument("--contact_match_weight", type=float, default=1.0)
  p.add_argument("--contact_match_beta", type=float, default=40.0)
  p.add_argument("--object_reward_mult", type=float, default=1.0)
  p.add_argument("--contact_miss_t", type=int, default=999999,
    help="Unused at eval (terminations wiped); needed only for build_env_cfg.")
  p.add_argument("--curriculum_scale", type=float, default=0.0,
    help="Unused at eval (curriculum unused post-build); needed for build_env_cfg.")
  p.add_argument("--obs_clip", type=float, default=5.0)
  p.add_argument("--num_envs_max", type=int, default=None,
    help="Cap on num_envs (= M); useful for smoke-testing on small N.")
  p.add_argument("--eval_seconds", type=float, default=60.0)
  p.add_argument("--threshold_ks", type=str, default="0.5,1.0,1.5,2.0,3.0")
  p.add_argument("--out_csv", required=True)
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  if len(args.base_checkpoints) != 1:
    p.error(f"--side {args.side} requires exactly 1 --base_checkpoints; got {len(args.base_checkpoints)}.")

  configure_torch_backends()
  torch.manual_seed(args.seed)
  device = "cuda:0"
  task_id = f"mjlab-maniptrans-{args.robot}-{args.side}"
  side_p = SIDE_PREFIX[args.side]

  ks = sorted(set([float(x) for x in args.threshold_ks.split(",")]))
  k_tensor = torch.tensor(ks, device=device)
  n_k = len(ks)

  # Inspect packed motion.pt for M and metadata.
  packed = torch.load(args.motion_file, weights_only=False)
  M = int(packed["motion_num_frames"].shape[0])
  motion_csv_index = packed.get("motion_csv_index")
  if motion_csv_index is None:
    raise RuntimeError(
      "motion.pt missing 'motion_csv_index' — required for the CSV index column."
    )
  motion_csv_index = motion_csv_index.tolist()
  motion_num_frames = packed["motion_num_frames"].tolist()

  if args.num_envs_max is not None:
    M = min(M, args.num_envs_max)
  print(f"[score_residual:{args.robot}] M={M}  eval_seconds={args.eval_seconds}", flush=True)

  # build_env_cfg from train_residual.py needs all the residual args.
  cfg_args = argparse.Namespace(
    robot=args.robot, side=args.side, motion_file=args.motion_file,
    index=None, num_envs=M,
    input_dir=args.input_dir,
    obj_density=args.obj_density,
    contact_match_weight=args.contact_match_weight,
    contact_match_beta=args.contact_match_beta,
    object_reward_mult=args.object_reward_mult,
    contact_miss_t=args.contact_miss_t,
    curriculum_scale=args.curriculum_scale,
  )
  cfg = build_env_cfg(cfg_args)

  motion_cmd_cfg = cfg.commands["motion"]
  assert isinstance(motion_cmd_cfg, ManipTransCommandCfg)
  motion_cmd_cfg.sampling_mode = "start"
  cfg.terminations = {}
  for group_name in ("actor", "critic"):
    if group_name in cfg.observations:
      cfg.observations[group_name].enable_corruption = False

  env = ManagerBasedRlEnv(cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env)
  cmd = cast(ManipTransCommand, env.command_manager.get_term("motion"))

  # Pin env i -> motion i (same monkey-patch trick as score_base_per_motion.py).
  orig_resample = cmd._resample_command
  saved_num_traj = cmd.motion.num_trajectories

  def _resample_pinned(env_ids):
    cmd.motion.num_trajectories = 1
    try:
      orig_resample(env_ids)
    finally:
      cmd.motion.num_trajectories = saved_num_traj

  cmd._resample_command = _resample_pinned  # type: ignore[method-assign]

  pinned_idxs = torch.arange(M, device=device, dtype=torch.long)
  cmd.env_traj_idx[:] = pinned_idxs

  obs, _ = wrapped.reset()
  cmd.env_traj_idx[:] = pinned_idxs

  # Build ResidualActor on top of frozen base.
  base_obs_dim, base_action_dim = _read_base_dims(args.base_checkpoints[0])
  agent_cfg = load_rl_cfg(task_id)
  agent_cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": args.residual_init_std,
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
  if args.obs_clip > 0:
    train_cfg["actor"]["obs_clip"] = args.obs_clip
    clipped_cls = "mjlab.tasks.maniptrans.rl.clipped_mlp_model.ClippedMLPModel"
    train_cfg["critic"]["class_name"] = clipped_cls
    train_cfg["critic"]["obs_clip"] = args.obs_clip

  runner = MjlabOnPolicyRunner(wrapped, train_cfg, ".", device)
  runner.load(args.residual_checkpoint)
  policy = runner.get_inference_policy(device=device)

  side_idx = 0  # right (or left, depending on --side); single-side so always 0
  pen_sensor = env.scene[f"{side_p}_fingertip_penetration"]
  force_sensor = env.scene[f"{side_p}_fingertip_contact"]

  n_steps = int(round(args.eval_seconds / env.step_dt))
  print(f"[score_residual:{args.robot}] step_dt={env.step_dt:.4f}  n_steps={n_steps}", flush=True)

  # Per-env (per-motion) accumulators.
  sum_wrist_pos = torch.zeros(M, device=device)
  sum_wrist_rot = torch.zeros(M, device=device)
  sum_tip_pos   = torch.zeros(M, device=device)  # mean across 5 fingers
  sum_obj_pos   = torch.zeros(M, device=device)
  sum_obj_rot   = torch.zeros(M, device=device)
  sum_joint     = torch.zeros(M, device=device)

  # Per-finger contact telemetry: 5 fingers per env.
  contact_ref_dist_sum = torch.zeros(M, 5, device=device)
  contact_pen_sum      = torch.zeros(M, 5, device=device)
  contact_force_sum    = torch.zeros(M, 5, device=device)
  contact_count_ref    = torch.zeros(M, 5, device=device)  # frames where ref_flag==1
  contact_count_contact = torch.zeros(M, 5, device=device)  # frames ref_flag==1 AND found>0

  count = 0
  t0 = time.time()
  with torch.no_grad():
    for step in range(n_steps):
      actions = policy(obs)
      obs, _, _, _ = wrapped.step(actions)

      # Errors (per-env scalars).
      wrist_pos_err = torch.norm(cmd.robot_wrist_pos_w - cmd.mano_wrist_pos_w, dim=-1)[:, side_idx]
      R_diff_wrist = (
        cmd.mano_wrist_rot_w[:, side_idx] @
        cmd.robot_wrist_rot_w[:, side_idx].transpose(-1, -2)
      )
      wrist_rot_err_deg = _rotmat_axis_angle_deg(R_diff_wrist)
      tip_err_per_finger = torch.norm(
        cmd.mano_tip_pos_w[:, side_idx] - cmd.robot_tip_pos_w[:, side_idx], dim=-1
      )  # (M, 5)
      tip_err_mean = tip_err_per_finger.mean(dim=-1)
      obj_pos_err = torch.norm(
        cmd.ref_obj_pos_w[:, side_idx] - cmd.sim_obj_pos_w[:, side_idx], dim=-1
      )
      sim_R = _quat_to_rotmat(cmd.sim_obj_quat_w[:, side_idx])
      R_diff_obj = cmd.ref_obj_rotmat_w[:, side_idx] @ sim_R.transpose(-1, -2)
      obj_rot_err_deg = _rotmat_axis_angle_deg(R_diff_obj)

      mano_l1 = cmd.mano_level_pos_w(args.side, 1)
      robot_l1 = cmd.robot_level_pos_w(args.side, 1)
      err_l1 = torch.norm(mano_l1 - robot_l1, dim=-1).mean(dim=-1)
      mano_l2 = cmd.mano_level_pos_w(args.side, 2)
      robot_l2 = cmd.robot_level_pos_w(args.side, 2)
      err_l2 = torch.norm(mano_l2 - robot_l2, dim=-1).mean(dim=-1)
      joint_err = 0.5 * (err_l1 + err_l2)

      sum_wrist_pos += wrist_pos_err
      sum_wrist_rot += wrist_rot_err_deg
      sum_tip_pos   += tip_err_mean
      sum_obj_pos   += obj_pos_err
      sum_obj_rot   += obj_rot_err_deg
      sum_joint     += joint_err

      # Per-finger contact (gated on ref_flag == 1; pen/force also gated on found).
      flag = cmd.ref_contact_flags[:, side_idx]                                 # (M, 5)
      found = (pen_sensor.data.found > 0).to(flag.dtype)                        # (M, 5)
      contact_gate = flag * found
      ref_dist = torch.norm(
        cmd.ref_contact_pos_w[:, side_idx] - cmd.robot_tip_pos_w[:, side_idx], dim=-1,
      )                                                                          # (M, 5)
      pen = torch.clamp(-pen_sensor.data.dist, min=0.0)                          # (M, 5)
      force = torch.norm(force_sensor.data.force, dim=-1)                        # (M, 5)
      contact_ref_dist_sum   += ref_dist * flag
      contact_pen_sum        += pen * contact_gate
      contact_force_sum      += force * contact_gate
      contact_count_ref      += flag
      contact_count_contact  += contact_gate

      count += 1
      if (step + 1) % 500 == 0 or step + 1 == n_steps:
        el = time.time() - t0
        rate = (step + 1) / max(el, 1e-9)
        eta = (n_steps - step - 1) / max(rate, 1e-9)
        print(f"[score_residual:{args.robot}] step {step+1}/{n_steps}  el={el:.1f}s eta={eta:.1f}s",
              flush=True)

  env.close()

  # Per-motion time-averaged errors.
  avg_wrist_pos = (sum_wrist_pos / count)             # m
  avg_wrist_rot = (sum_wrist_rot / count)             # deg
  avg_tip_pos   = (sum_tip_pos   / count)             # m
  avg_obj_pos   = (sum_obj_pos   / count)             # m
  avg_obj_rot   = (sum_obj_rot   / count)             # deg
  avg_joint     = (sum_joint     / count)             # m

  # Per-motion SR@k boolean: pass if ALL of obj_pos / tip_pos / obj_rot / joint_pos under k * threshold.
  # Shapes: avg_*[:, None] vs k_tensor[None, :] -> (M, n_k).
  fail_obj_pos = avg_obj_pos.unsqueeze(-1) > (k_tensor * TH_OBJ_POS)
  fail_tip_pos = avg_tip_pos.unsqueeze(-1) > (k_tensor * TH_TIP_POS)
  fail_obj_rot = avg_obj_rot.unsqueeze(-1) > (k_tensor * TH_OBJ_ROT)
  fail_joint   = avg_joint.unsqueeze(-1)   > (k_tensor * TH_JOINT_POS)
  pass_mask = ~(fail_obj_pos | fail_tip_pos | fail_obj_rot | fail_joint)  # (M, n_k)

  # Per-finger contact stats — same gating as score_perside.py.
  count_ref_safe     = contact_count_ref.clamp(min=1.0)
  count_contact_safe = contact_count_contact.clamp(min=1.0)
  cm_ref_dist = (contact_ref_dist_sum / count_ref_safe).cpu()    # (M, 5)
  cm_found    = (contact_count_contact / count_ref_safe).cpu()   # hit rate
  cm_pen      = (contact_pen_sum / count_contact_safe).cpu()
  cm_force    = (contact_force_sum / count_contact_safe).cpu()

  # CSV write.
  out_path = Path(args.out_csv)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  pass_cols = [f"pass@{k:.1f}" for k in ks]
  contact_cols = []
  for kind in ("ref_dist", "pen", "force", "found"):
    for f in FINGER_NAMES:
      contact_cols.append(f"contact_{kind}_{side_p}_{f}")
  header = (
    ["index", "num_frame",
     f"error_wrist_pos_{side_p}", f"error_wrist_rot_{side_p}",
     f"error_tip_pos_{side_p}",
     f"error_obj_pos_{side_p}", f"error_obj_rot_{side_p}",
     f"error_joint_{side_p}"]
    + pass_cols + contact_cols
  )

  avg_wrist_pos_l = avg_wrist_pos.cpu().tolist()
  avg_wrist_rot_l = avg_wrist_rot.cpu().tolist()
  avg_tip_pos_l   = avg_tip_pos.cpu().tolist()
  avg_obj_pos_l   = avg_obj_pos.cpu().tolist()
  avg_obj_rot_l   = avg_obj_rot.cpu().tolist()
  avg_joint_l     = avg_joint.cpu().tolist()
  pass_mask_l     = pass_mask.cpu().tolist()

  # Per-motion summary line for the log.
  sr_per_k = pass_mask.float().mean(dim=0).cpu().tolist()
  print("=" * 72)
  print(f"RESIDUAL: {args.residual_checkpoint}")
  print(f"BASE: {args.base_checkpoints[0]}  (scale={args.residual_action_scale})")
  print(f"per-motion eval: M={M}  eval_seconds={args.eval_seconds}  "
        f"step_dt={env.step_dt:.4f}  n_steps={n_steps}")
  for i, k in enumerate(ks):
    flag = "  <- paper strict" if k == 1.0 else ""
    print(f"  SR @ k={k:.1f}:  {sr_per_k[i]*100:6.2f}%   "
          f"(obj={k * TH_OBJ_POS * 100:.1f}cm  tip={k * TH_TIP_POS * 100:.1f}cm  "
          f"rot={k * TH_OBJ_ROT:.0f}deg  joint={k * TH_JOINT_POS * 100:.1f}cm){flag}")
  print(f"-> CSV: {out_path}")
  print("=" * 72)

  with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for i in range(M):
      row = [
        int(motion_csv_index[i]),
        int(motion_num_frames[i]),
        float(avg_wrist_pos_l[i]) * 100.0,  # cm
        float(avg_wrist_rot_l[i]),          # deg
        float(avg_tip_pos_l[i])   * 100.0,  # cm
        float(avg_obj_pos_l[i])   * 100.0,  # cm
        float(avg_obj_rot_l[i]),            # deg
        float(avg_joint_l[i])     * 100.0,  # cm
      ]
      row += [int(bool(pass_mask_l[i][j])) for j in range(n_k)]
      for kind_arr in (cm_ref_dist, cm_pen, cm_force, cm_found):
        row += [float(kind_arr[i, fi].item()) for fi in range(5)]
      w.writerow(row)

  el = time.time() - t0
  print(f"[score_residual:{args.robot}] DONE  rows={M}  wall={el:.1f}s")


if __name__ == "__main__":
  main()
