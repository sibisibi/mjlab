"""Per-side / two-base bimanual rollout video renderer for ManipTrans residual policies.

Single-hand (--side {right,left}, ONE per-side base ckpt) and bimanual two-base
(--side bimanual, TWO per-side base ckpts) modes via --base_checkpoints (nargs
1 or 2). The "render two bases alone" mode is out of scope here — use
.progress/10-mirror-smoke-test/src/render_bimanual_compose.py.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import cast

import imageio
import torch

import mjlab.tasks.maniptrans.config  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.maniptrans.mdp import ManipTransCommandCfg
from mjlab.tasks.maniptrans.mdp.commands import ManipTransCommand
from mjlab.tasks.maniptrans.scripts.train_residual import build_env_cfg
from mjlab.tasks.registry import load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViewerConfig


GLOBAL_HAND_OBS_TERMS = {"wrist_state"}


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
  p.add_argument("--motion_file", required=True)
  p.add_argument("--data_dir", required=True)
  p.add_argument("--pool_dir", required=True)
  p.add_argument("--obj_density", type=float, default=800.0)
  p.add_argument("--contact_match_weight", type=float, default=1.0)
  p.add_argument("--contact_match_beta", type=float, default=40.0)
  p.add_argument("--contact_miss_t", type=int, default=999999)
  p.add_argument("--object_reward_mult", type=float, default=1.0)
  p.add_argument("--obs_clip", type=float, default=0.0)
  p.add_argument("--base_checkpoints", nargs="+", required=True,
    help="1 path for --side {right,left}; 2 paths [right, left] for --side bimanual.")
  p.add_argument("--checkpoint", required=True, help="Stage 2 residual checkpoint.")
  p.add_argument("--residual_action_scale", type=float, default=1.0)
  p.add_argument("--init_std", type=float, default=0.37)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--output", required=True, help="Output mp4 path.")
  p.add_argument("--max_steps", type=int, default=-1,
    help="-1 auto-derives motion_T-1 from motion file.")
  p.add_argument("--fps", type=float, default=-1.0,
    help="-1 auto-derives policy rate (1/env.step_dt).")
  p.add_argument("--azimuth", type=float, default=90.0)
  p.add_argument("--elevation", type=float, default=-25.0)
  p.add_argument("--distance", type=float, default=0.8)
  p.add_argument("--num_envs", type=int, default=1)
  p.add_argument("--trace_metrics", action="store_true")
  p.add_argument("--grace_steps", type=int, default=15)
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

  cfg = build_env_cfg(args)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, ManipTransCommandCfg)
  motion_cmd.sampling_mode = "start"
  cfg.terminations = {}

  for group_name in ("actor", "critic"):
    if group_name in cfg.observations:
      cfg.observations[group_name].enable_corruption = False

  cfg.viewer = ViewerConfig(
    width=1280, height=960,
    distance=args.distance, elevation=args.elevation, azimuth=args.azimuth,
    lookat=(0.0, 0.0, 0.3),
    origin_type=ViewerConfig.OriginType.WORLD,
  )

  env = ManagerBasedRlEnv(cfg, device=device, render_mode="rgb_array")
  wrapped = RslRlVecEnvWrapper(env)

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

  obs, _ = wrapped.reset()

  agent_cfg = load_rl_cfg(task_id)
  agent_cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": args.init_std,
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
  runner.load(args.checkpoint)
  policy = runner.get_inference_policy(device=device)

  frames = []
  cmd = cast(ManipTransCommand, env.command_manager.get_term("motion"))
  motion_T = int(cmd.motion.time_step_totals.max().item())
  if args.max_steps <= 0:
    args.max_steps = max(1, motion_T - 1)
  if args.fps <= 0:
    args.fps = 1.0 / env.step_dt
  print(f"[rollout] motion_T={motion_T}  max_steps={args.max_steps}  fps={args.fps:.3f}")

  trace_tip_max: list[float] = []
  trace_tip_mean: list[float] = []
  trace_obj_pos: list[float] = []
  trace_obj_rot: list[float] = []
  first_fail: dict[str, int | None] = {"obj_pos_3cm": None, "tip_6cm": None, "obj_rot_30": None}

  for step in range(args.max_steps):
    with torch.no_grad():
      actions = policy(obs)
    obs, _, _, _ = wrapped.step(actions)
    img = env.render()
    if img is not None:
      frames.append(img)

    if args.trace_metrics:
      tip_err_per_finger = torch.norm(cmd.mano_tip_pos_w - cmd.robot_tip_pos_w, dim=-1)
      tip_err_max_side = tip_err_per_finger.max(dim=-1).values.max(dim=-1).values
      tip_err_mean_side = tip_err_per_finger.mean(dim=-1).mean(dim=-1)
      obj_pos_err = torch.norm(cmd.ref_obj_pos_w - cmd.sim_obj_pos_w, dim=-1).max(dim=-1).values
      sim_R = _quat_to_rotmat(cmd.sim_obj_quat_w)
      R_diff = cmd.ref_obj_rotmat_w @ sim_R.transpose(-1, -2)
      obj_rot_err_deg = _rotmat_axis_angle_deg(R_diff).max(dim=-1).values

      t_max = float(tip_err_max_side.mean().item()) * 100.0
      t_mean = float(tip_err_mean_side.mean().item()) * 100.0
      p_obj = float(obj_pos_err.mean().item()) * 100.0
      r_obj = float(obj_rot_err_deg.mean().item())
      trace_tip_max.append(t_max)
      trace_tip_mean.append(t_mean)
      trace_obj_pos.append(p_obj)
      trace_obj_rot.append(r_obj)

      if step >= args.grace_steps:
        if first_fail["obj_pos_3cm"] is None and p_obj > 3.0:
          first_fail["obj_pos_3cm"] = step
        if first_fail["tip_6cm"] is None and t_max > 6.0:
          first_fail["tip_6cm"] = step
        if first_fail["obj_rot_30"] is None and r_obj > 30.0:
          first_fail["obj_rot_30"] = step

  env.close()

  out_path = Path(args.output)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimsave(str(out_path), frames, fps=args.fps)
  print(f"Saved {len(frames)} frames ({args.fps} fps) to {out_path}")

  if args.trace_metrics and trace_tip_max:
    import statistics
    def mn(x): return statistics.mean(x)
    def mx(x): return max(x)
    print("=" * 60)
    print(f"TRACE (num_envs={args.num_envs}, T={len(trace_tip_max)} steps)")
    print("-" * 60)
    print(f"  tip max  (per frame max over fingers/sides):  mean={mn(trace_tip_max):5.2f} cm  peak={mx(trace_tip_max):5.2f} cm")
    print(f"  tip mean (per frame mean over fingers/sides): mean={mn(trace_tip_mean):5.2f} cm  peak={mx(trace_tip_mean):5.2f} cm")
    print(f"  obj_pos  (max over sides)                   : mean={mn(trace_obj_pos):5.2f} cm  peak={mx(trace_obj_pos):5.2f} cm")
    print(f"  obj_rot  (max over sides)                   : mean={mn(trace_obj_rot):5.2f} deg peak={mx(trace_obj_rot):5.2f} deg")
    print("-" * 60)
    print(f"  First frame that crosses k=1.0 threshold (grace={args.grace_steps}):")
    for k, v in first_fail.items():
      print(f"    {k:15s}: {'frame ' + str(v) if v is not None else 'never crossed (SURVIVED)'}")
    print("=" * 60)


if __name__ == "__main__":
  main()
