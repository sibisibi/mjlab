"""Per-motion scorer for ManipTrans Stage-1 base policies.

Pins env i to motion i in the packed motion file (M motions => M envs), runs
each env for `--eval_seconds` of simulated time (short motions auto-cycle via
the motion command's wrap-around), and emits per-env time-averaged + per-env
max metrics to a CSV row.

Output CSV columns (25 total):
  index, num_frame
  + 12 errors:   error_wrist_pos_r, error_wrist_rot_r, error_wrist_vel_r,
                 error_wrist_angvel_r, error_tip_pos_r_{thumb,index,middle,
                 ring,pinky}, error_level1_r, error_level2_r,
                 error_joints_vel_r
  + 6 vel stats: mean_wrist_vel, max_wrist_vel,
                 mean_wrist_angvel, max_wrist_angvel,
                 mean_joints_vel, max_joints_vel
  + 5 max-tip:   max_tip_pos_r_{thumb,index,middle,ring,pinky}

Counterpart to `score_perside.py` for residual; "score base alone" is no
longer out of scope (this is it). Used by `scripts/eval_base.sh`.
"""

from __future__ import annotations

import argparse
import csv
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
from mjlab.tasks.maniptrans.scripts.train_base import build_env_cfg
from mjlab.tasks.registry import load_rl_cfg
from mjlab.utils.torch import configure_torch_backends


METRIC_KEYS = (
  "error_wrist_pos_r",
  "error_wrist_rot_r",
  "error_wrist_vel_r",
  "error_wrist_angvel_r",
  "error_tip_pos_r_thumb",
  "error_tip_pos_r_index",
  "error_tip_pos_r_middle",
  "error_tip_pos_r_ring",
  "error_tip_pos_r_pinky",
  "error_level1_r",
  "error_level2_r",
  "error_joints_vel_r",
)

TIP_KEYS = (
  "error_tip_pos_r_thumb",
  "error_tip_pos_r_index",
  "error_tip_pos_r_middle",
  "error_tip_pos_r_ring",
  "error_tip_pos_r_pinky",
)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--robot", required=True)
  p.add_argument("--side", default="right", choices=["right", "left", "bimanual"])
  p.add_argument("--motion_file", required=True,
    help="Packed motion .pt produced by package_motion.py.")
  p.add_argument("--checkpoint", required=True, help="Stage-1 base checkpoint.")
  p.add_argument("--out_csv", required=True)
  p.add_argument("--eval_seconds", type=float, default=60.0)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--obs_clip", type=float, default=5.0)
  p.add_argument("--num_envs_max", type=int, default=None,
    help="Cap on num_envs (= M); useful for smoke-testing on a small N.")
  args = p.parse_args()

  configure_torch_backends()
  torch.manual_seed(args.seed)
  device = "cuda:0"
  task_id = f"mjlab-maniptrans-{args.robot}-{args.side}"

  # Inspect packed motion.pt to determine M and pull metadata.
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
  print(f"[score_base:{args.robot}] M={M}  eval_seconds={args.eval_seconds}", flush=True)

  cfg_args = argparse.Namespace(
    robot=args.robot, side=args.side, motion_file=args.motion_file,
    index=None, num_envs=M,
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

  # Pin env i -> motion i. Monkey-patch _resample_command via the
  # num_trajectories=1 flip trick: the original would randomize env_traj_idx
  # at commands.py:1069-1073, but that branch is gated on num_trajectories>1.
  # Flipping to 1 around the call skips randomization while still letting the
  # original do the warm-start joint-state writes for the (pinned) motion.
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
  cmd.env_traj_idx[:] = pinned_idxs  # re-pin in case any inner reset path touched it

  agent_cfg = load_rl_cfg(task_id)
  train_cfg = asdict(agent_cfg)
  if args.obs_clip > 0:
    clipped_cls = "mjlab.tasks.maniptrans.rl.clipped_mlp_model.ClippedMLPModel"
    train_cfg["actor"]["class_name"] = clipped_cls
    train_cfg["actor"]["obs_clip"] = args.obs_clip
    train_cfg["critic"]["class_name"] = clipped_cls
    train_cfg["critic"]["obs_clip"] = args.obs_clip
  runner = MjlabOnPolicyRunner(wrapped, train_cfg, ".", device)
  runner.load(args.checkpoint)
  policy = runner.get_inference_policy(device=device)

  n_steps = int(round(args.eval_seconds / env.step_dt))
  print(f"[score_base:{args.robot}] step_dt={env.step_dt:.4f}  n_steps={n_steps}", flush=True)

  # Per-env accumulators. Errors via cmd.metrics; vel stats via cmd.robot_*.
  accum = {k: torch.zeros(M, device=device) for k in METRIC_KEYS}
  max_tip = {k: torch.zeros(M, device=device) for k in TIP_KEYS}
  side_idx = 0  # right
  side_for_joints = "right" if args.side == "right" else ("left" if args.side == "left" else "right")
  sum_wv = torch.zeros(M, device=device)
  max_wv = torch.zeros(M, device=device)
  sum_wav = torch.zeros(M, device=device)
  max_wav = torch.zeros(M, device=device)
  sum_jv = torch.zeros(M, device=device)
  max_jv = torch.zeros(M, device=device)

  t0 = time.time()
  for step in range(n_steps):
    with torch.no_grad():
      actions = policy(obs)
    obs, _, _, _ = wrapped.step(actions)
    for k in METRIC_KEYS:
      accum[k] += cmd.metrics[k]
    for k in TIP_KEYS:
      max_tip[k] = torch.maximum(max_tip[k], cmd.metrics[k])
    wv = cmd.robot_wrist_vel_w[:, side_idx].abs().mean(dim=-1)
    wav = cmd.robot_wrist_angvel_w[:, side_idx].abs().mean(dim=-1)
    jv = cmd.robot_all_joints_vel_w(side_for_joints).abs().mean(dim=-1).mean(dim=-1)
    sum_wv += wv; sum_wav += wav; sum_jv += jv
    max_wv = torch.maximum(max_wv, wv)
    max_wav = torch.maximum(max_wav, wav)
    max_jv = torch.maximum(max_jv, jv)
    if (step + 1) % 500 == 0 or step + 1 == n_steps:
      el = time.time() - t0
      rate = (step + 1) / max(el, 1e-9)
      eta = (n_steps - step - 1) / max(rate, 1e-9)
      print(f"[score_base:{args.robot}] step {step+1}/{n_steps}  el={el:.1f}s eta={eta:.1f}s",
            flush=True)

  env.close()

  avgs = {k: (accum[k] / n_steps).cpu().tolist() for k in METRIC_KEYS}
  max_tip_l = {k: max_tip[k].cpu().tolist() for k in TIP_KEYS}
  mean_wv = (sum_wv / n_steps).cpu().tolist()
  mean_wav = (sum_wav / n_steps).cpu().tolist()
  mean_jv = (sum_jv / n_steps).cpu().tolist()
  max_wv_l = max_wv.cpu().tolist()
  max_wav_l = max_wav.cpu().tolist()
  max_jv_l = max_jv.cpu().tolist()

  out_path = Path(args.out_csv)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  vel_cols = [
    "mean_wrist_vel", "max_wrist_vel",
    "mean_wrist_angvel", "max_wrist_angvel",
    "mean_joints_vel", "max_joints_vel",
  ]
  max_tip_cols = [
    "max_tip_pos_r_thumb", "max_tip_pos_r_index", "max_tip_pos_r_middle",
    "max_tip_pos_r_ring", "max_tip_pos_r_pinky",
  ]
  header = ["index", "num_frame"] + list(METRIC_KEYS) + vel_cols + max_tip_cols
  with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for i in range(M):
      row = [
        int(motion_csv_index[i]),
        int(motion_num_frames[i]),
      ] + [float(avgs[k][i]) for k in METRIC_KEYS] + [
        float(mean_wv[i]),  float(max_wv_l[i]),
        float(mean_wav[i]), float(max_wav_l[i]),
        float(mean_jv[i]),  float(max_jv_l[i]),
      ] + [float(max_tip_l[k][i]) for k in TIP_KEYS]
      w.writerow(row)

  el = time.time() - t0
  print(f"[score_base:{args.robot}] DONE  rows={M}  wall={el:.1f}s  -> {out_path}")


if __name__ == "__main__":
  main()
