"""Per-side / two-base bimanual scorer for ManipTrans residual policies.

Single-hand mode (--side {right,left}, ONE per-side base ckpt) and bimanual
two-base mode (--side bimanual, TWO per-side base ckpts [right, left]) are
both supported via --base_checkpoints (nargs 1 or 2). --residual_checkpoint
is required; "score base alone" is out of scope.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from typing import cast

import torch

import mjlab.tasks.maniptrans.config  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.maniptrans.mdp import ManipTransCommandCfg
from mjlab.tasks.maniptrans.mdp.commands import ManipTransCommand
from mjlab.tasks.maniptrans.scripts.train_residual import build_env_cfg
from mjlab.tasks.registry import load_rl_cfg
from mjlab.utils.torch import configure_torch_backends


GLOBAL_HAND_OBS_TERMS = {"wrist_state"}
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
SIDE_PREFIX = {"right": "r", "left": "l"}


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


def _read_base_dims(path: str) -> tuple[int, int]:
  ck = torch.load(path, map_location="cpu", weights_only=False)
  sd = ck["actor_state_dict"]
  obs_d = int(sd["obs_normalizer._mean"].shape[-1])
  last_idx = max(int(k.split(".")[1]) for k in sd if k.startswith("mlp."))
  act_d = int(sd[f"mlp.{last_idx}.bias"].shape[0])
  return obs_d, act_d


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--robot", required=True)
  p.add_argument("--side", required=True, choices=["right", "left", "bimanual"])
  p.add_argument("--motion_file", required=True,
    help="Packed motion .pt produced by package_motion_batch.py.")
  p.add_argument("--index", type=int, default=None,
    help="Optional: select one motion (by row index) inside the packed .pt.")
  p.add_argument("--input_dir", required=True,
    help="Pool root for object meshes; resolves <input_dir>/<pool_rel_dir>/<mesh_rel>.")
  p.add_argument("--obj_density", type=float, default=800.0)
  p.add_argument("--contact_match_weight", type=float, default=1.0)
  p.add_argument("--contact_match_beta", type=float, default=40.0)
  p.add_argument("--contact_miss_t", type=int, default=999999,
    help="Unused at eval (cfg.terminations wiped post-build); needed only "
         "to satisfy build_env_cfg's termination registration.")
  p.add_argument("--object_reward_mult", type=float, default=1.0)
  p.add_argument("--curriculum_scale", type=float, default=0.0,
    help="Unused at eval (cfg.curriculum unused post-build); needed for build_env_cfg.")
  p.add_argument("--obs_clip", type=float, default=5.0)
  p.add_argument("--base_checkpoints", nargs="+", required=True,
    help="1 path for --side {right,left}; 2 paths [right, left] for --side bimanual.")
  p.add_argument("--residual_checkpoint", required=True)
  p.add_argument("--residual_action_scale", type=float, default=1.0)
  p.add_argument("--residual_init_std", type=float, default=0.37)
  p.add_argument("--num_envs", type=int, default=4096)
  p.add_argument("--grace_steps", type=int, default=15)
  p.add_argument("--tail_skip_steps", type=int, default=0)
  p.add_argument("--threshold_ks", type=str, default="0.5,1.0,1.5,2.0,3.0")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  base_ckpts = list(args.base_checkpoints)
  if args.side == "bimanual" and len(base_ckpts) != 2:
    p.error(f"--side bimanual requires --base_checkpoints with 2 paths [right, left]; got {len(base_ckpts)}.")
  if args.side in ("right", "left") and len(base_ckpts) != 1:
    p.error(f"--side {args.side} requires --base_checkpoints with 1 path; got {len(base_ckpts)}.")

  configure_torch_backends()
  torch.manual_seed(args.seed)
  device = "cuda:0"
  task_id = f"mjlab-maniptrans-{args.robot}-{args.side}"

  ks = sorted(set([float(x) for x in args.threshold_ks.split(",")] + [1.0]))
  n_k = len(ks)
  k1_idx = ks.index(1.0)
  k_tensor = torch.tensor(ks, device=device)

  cfg = build_env_cfg(args)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, ManipTransCommandCfg)
  motion_cmd.sampling_mode = "start"
  cfg.terminations = {}

  for group_name in ("actor", "critic"):
    if group_name in cfg.observations:
      cfg.observations[group_name].enable_corruption = False

  env = ManagerBasedRlEnv(cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env)

  agent_cfg = load_rl_cfg(task_id)

  base_dims = [_read_base_dims(p) for p in base_ckpts]
  if len(base_dims) == 2 and base_dims[0] != base_dims[1]:
    raise ValueError(f"Per-side bases disagree on dims: right={base_dims[0]} vs left={base_dims[1]}.")
  base_obs_dim, base_action_dim = base_dims[0]

  perside_term_specs = None
  n_wrist_per_side = 0
  if len(base_ckpts) == 2:
    active = env.observation_manager.active_terms["actor"]
    shapes = env.observation_manager.group_obs_term_dim["actor"]
    perside_term_specs = []
    accum = 0
    for name, shape in zip(active, shapes):
      bim_dim = int(shape[-1]) if isinstance(shape, (tuple, list)) else int(shape)
      is_global = name in GLOBAL_HAND_OBS_TERMS
      per_side_dim = bim_dim if is_global else (bim_dim // 2)
      if not is_global and bim_dim % 2 != 0:
        raise RuntimeError(f"Per-side hand obs term {name!r} has odd bimanual dim {bim_dim}.")
      perside_term_specs.append((per_side_dim, is_global))
      accum += per_side_dim
      if accum == base_obs_dim:
        break
      if accum > base_obs_dim:
        raise RuntimeError(f"Per-side accum {accum} overshot base_obs_dim {base_obs_dim} at {name!r}.")
    if accum != base_obs_dim:
      raise RuntimeError(f"Per-side prefix sum {accum} != base_obs_dim {base_obs_dim}.")
    bimanual_n_wrist = env.action_manager.get_term("maniptrans")._n_wrist
    if bimanual_n_wrist % 2 != 0:
      raise RuntimeError(f"Bimanual n_wrist={bimanual_n_wrist} is not even.")
    n_wrist_per_side = bimanual_n_wrist // 2

  agent_cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": args.residual_init_std,
    "std_type": "scalar",
  }
  train_cfg = asdict(agent_cfg)
  train_cfg["actor"]["class_name"] = (
    "mjlab.tasks.maniptrans.rl.residual_actor.ResidualActor"
  )
  train_cfg["actor"]["base_checkpoints"] = base_ckpts
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

  runner = MjlabOnPolicyRunner(wrapped, train_cfg, ".", device)
  runner.load(args.residual_checkpoint)
  policy = runner.get_inference_policy(device=device)

  cmd = cast(ManipTransCommand, env.command_manager.get_term("motion"))
  motion_T = int(cmd.motion.time_step_totals.max().item())
  n_sides = len(cmd._side_list)
  T = max(1, motion_T - 1)
  eval_end = max(args.grace_steps + 1, T - args.tail_skip_steps)
  N = args.num_envs
  print(f"[scorer:RESIDUAL] motion_T={motion_T}  n_envs={N}  n_sides={n_sides}  "
        f"T_rollout={T}  eval_window=[{args.grace_steps},{eval_end})  "
        f"pin_objects={motion_cmd.pin_objects}  "
        f"ckpt={args.residual_checkpoint}")

  obs, _ = wrapped.reset()

  fail_mask = torch.zeros(N, n_k, dtype=torch.bool, device=device)
  sum_wrist_pos = torch.zeros(N, n_sides, device=device)
  sum_wrist_rot = torch.zeros(N, n_sides, device=device)
  sum_tip_pos   = torch.zeros(N, n_sides, device=device)
  sum_obj_pos   = torch.zeros(N, n_sides, device=device)
  sum_obj_rot   = torch.zeros(N, n_sides, device=device)
  sum_joint     = torch.zeros(N, n_sides, device=device)
  count = 0

  # Per-finger contact telemetry. Two gates per (side, finger):
  #   - ref_flag==1            → denominator for found (hit rate)
  #   - ref_flag==1 AND force>0 → denominator for ref_dist/pen/force
  contact_ref_dist_sum     = torch.zeros(N, n_sides, 5, device=device)
  contact_pen_sum          = torch.zeros(N, n_sides, 5, device=device)
  contact_force_sum        = torch.zeros(N, n_sides, 5, device=device)
  contact_count_ref        = torch.zeros(N, n_sides, 5, device=device)
  contact_count_contact    = torch.zeros(N, n_sides, 5, device=device)
  pen_sensors = {
    side: env.scene[f"{SIDE_PREFIX[side]}_fingertip_penetration"]
    for side in cmd._side_list
  }
  force_sensors = {
    side: env.scene[f"{SIDE_PREFIX[side]}_fingertip_contact"]
    for side in cmd._side_list
  }

  TH_OBJ_POS = 0.03
  TH_TIP_POS = 0.06
  TH_OBJ_ROT = 30.0
  TH_JOINT_POS = 0.08

  with torch.no_grad():
    for t in range(T):
      actions = policy(obs)
      obs, _, _, _ = wrapped.step(actions)

      wrist_pos_err = torch.norm(cmd.robot_wrist_pos_w - cmd.mano_wrist_pos_w, dim=-1)
      R_diff_wrist = cmd.mano_wrist_rot_w @ cmd.robot_wrist_rot_w.transpose(-1, -2)
      wrist_rot_err_deg = _rotmat_axis_angle_deg(R_diff_wrist)
      tip_err_per_finger = torch.norm(cmd.mano_tip_pos_w - cmd.robot_tip_pos_w, dim=-1)
      tip_err_mean = tip_err_per_finger.mean(dim=-1)
      obj_pos_err = torch.norm(cmd.ref_obj_pos_w - cmd.sim_obj_pos_w, dim=-1)
      sim_R = _quat_to_rotmat(cmd.sim_obj_quat_w)
      R_diff_obj = cmd.ref_obj_rotmat_w @ sim_R.transpose(-1, -2)
      obj_rot_err_deg = _rotmat_axis_angle_deg(R_diff_obj)

      joint_err = torch.zeros(N, n_sides, device=device)
      for si, side in enumerate(cmd._side_list):
        mano_l1 = cmd.mano_level_pos_w(side, 1)
        robot_l1 = cmd.robot_level_pos_w(side, 1)
        err_l1 = torch.norm(mano_l1 - robot_l1, dim=-1).mean(dim=-1)
        mano_l2 = cmd.mano_level_pos_w(side, 2)
        robot_l2 = cmd.robot_level_pos_w(side, 2)
        err_l2 = torch.norm(mano_l2 - robot_l2, dim=-1).mean(dim=-1)
        joint_err[:, si] = 0.5 * (err_l1 + err_l2)

      if args.grace_steps <= t < eval_end:
        sum_wrist_pos += wrist_pos_err
        sum_wrist_rot += wrist_rot_err_deg
        sum_tip_pos   += tip_err_mean
        sum_obj_pos   += obj_pos_err
        sum_obj_rot   += obj_rot_err_deg
        sum_joint     += joint_err
        count += 1

        for si, side in enumerate(cmd._side_list):
          pen_sens = pen_sensors[side]
          force_sens = force_sensors[side]
          flag = cmd.ref_contact_flags[:, si]                                       # (N, 5)
          found = (pen_sens.data.found > 0).to(flag.dtype)                           # (N, 5)
          contact_gate = flag * found
          ref_dist = torch.norm(
            cmd.ref_contact_pos_w[:, si] - cmd.robot_tip_pos_w[:, si], dim=-1,
          )                                                                          # (N, 5)
          pen = torch.clamp(-pen_sens.data.dist, min=0.0)                            # (N, 5)
          force = torch.norm(force_sens.data.force, dim=-1)                          # (N, 5)
          contact_ref_dist_sum[:, si]  += ref_dist * flag
          contact_pen_sum[:, si]       += pen * contact_gate
          contact_force_sum[:, si]     += force * contact_gate
          contact_count_ref[:, si]     += flag
          contact_count_contact[:, si] += contact_gate

  per_env_tip_avg     = (sum_tip_pos   / count).max(dim=-1).values
  per_env_obj_pos_avg = (sum_obj_pos   / count).max(dim=-1).values
  per_env_obj_rot_avg = (sum_obj_rot   / count).max(dim=-1).values
  per_env_joint_avg   = (sum_joint     / count).max(dim=-1).values

  fail_obj_pos = per_env_obj_pos_avg.unsqueeze(-1) > (k_tensor * TH_OBJ_POS)
  fail_tip_pos = per_env_tip_avg.unsqueeze(-1)     > (k_tensor * TH_TIP_POS)
  fail_obj_rot = per_env_obj_rot_avg.unsqueeze(-1) > (k_tensor * TH_OBJ_ROT)
  fail_joint   = per_env_joint_avg.unsqueeze(-1)   > (k_tensor * TH_JOINT_POS)
  fail_mask = fail_obj_pos | fail_tip_pos | fail_obj_rot | fail_joint

  success_mask = ~fail_mask
  success_rate = success_mask.float().mean(dim=0)
  k1_mask = success_mask[:, k1_idx]
  n_k1 = int(k1_mask.sum().item())

  def _reduce(acc: torch.Tensor) -> float:
    return float((acc / count).mean(dim=-1).mean().item())

  E_t   = _reduce(sum_wrist_pos) * 100.0
  E_r   = _reduce(sum_wrist_rot)
  E_ft  = _reduce(sum_tip_pos)   * 100.0
  E_op  = _reduce(sum_obj_pos)   * 100.0
  E_or  = _reduce(sum_obj_rot)
  E_j   = _reduce(sum_joint)     * 100.0

  # Per-finger conditional contact metrics, matching training-time keys.
  # ref_dist + found gated on ref_flag==1; pen + force gated on ref_flag==1 AND found.
  contact_total_sum_rd   = contact_ref_dist_sum.sum(dim=0)  # (n_sides, 5)
  contact_total_sum_pen  = contact_pen_sum.sum(dim=0)
  contact_total_sum_f    = contact_force_sum.sum(dim=0)
  contact_total_count_r  = contact_count_ref.sum(dim=0).clamp(min=1.0)
  contact_total_count_c  = contact_count_contact.sum(dim=0).clamp(min=1.0)
  cm_ref_dist = (contact_total_sum_rd / contact_total_count_r).cpu()
  cm_found    = (contact_count_contact.sum(dim=0) / contact_total_count_r).cpu()
  cm_pen      = (contact_total_sum_pen / contact_total_count_c).cpu()
  cm_force    = (contact_total_sum_f / contact_total_count_c).cpu()

  print("=" * 72)
  print(f"RESIDUAL checkpoint: {args.residual_checkpoint}")
  print(f"frozen base(s): {base_ckpts}  (scale={args.residual_action_scale})")
  print(f"rollout     : motion_T={motion_T}  T={T}  N={N}  "
        f"window=[{args.grace_steps},{eval_end})  pin_objects={motion_cmd.pin_objects}")
  print("-" * 72)
  for i, k in enumerate(ks):
    sr = float(success_rate[i].item()) * 100.0
    flag = "  <- paper strict" if k == 1.0 else ""
    print(f"  SR @ k={k:.1f}:  {sr:6.2f}%   "
          f"(obj={k * TH_OBJ_POS * 100:.1f}cm  tip={k * TH_TIP_POS * 100:.1f}cm  "
          f"rot={k * TH_OBJ_ROT:.0f}deg  joint={k * TH_JOINT_POS * 100:.1f}cm){flag}")
  print("-" * 72)
  print(f"Errors averaged over ALL {N} envs  ({n_k1}/{N} passed @ k=1.0):")
  print(f"  E_t  (wrist pos)    : {E_t:7.2f} cm")
  print(f"  E_r  (wrist rot)    : {E_r:7.2f} deg")
  print(f"  E_ft (fingertip pos): {E_ft:7.2f} cm")
  print(f"  E_j  (joint pos)    : {E_j:7.2f} cm")
  print(f"  E_obj_pos           : {E_op:7.2f} cm")
  print(f"  E_obj_rot           : {E_or:7.2f} deg")
  print("-" * 72)
  print("Per-finger contact telemetry (mean over ref_flag==1 frames, count-weighted):")
  for si, side in enumerate(cmd._side_list):
    p = SIDE_PREFIX[side]
    for fi, finger in enumerate(FINGER_NAMES):
      print(
        f"  contact_ref_dist_{p}_{finger:<6s}: {cm_ref_dist[si, fi].item():.6f} m  "
        f"contact_pen_{p}_{finger:<6s}: {cm_pen[si, fi].item():.6f} m  "
        f"contact_force_{p}_{finger:<6s}: {cm_force[si, fi].item():.4f} N  "
        f"contact_found_{p}_{finger:<6s}: {cm_found[si, fi].item():.4f}"
      )
  print("=" * 72)

  env.close()


if __name__ == "__main__":
  main()
