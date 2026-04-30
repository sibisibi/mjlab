"""Per-side base rollout video renderer for ManipTrans Stage 1 base policies.

Mirrors rollout_perside.py but loads only a Stage 1 base checkpoint (no
ResidualActor). Used to inspect a multi-traj base policy on a single
reference trajectory.
"""

from __future__ import annotations

import argparse
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
from mjlab.tasks.maniptrans.scripts.train_base import build_env_cfg
from mjlab.tasks.registry import load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViewerConfig


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--robot", required=True)
  p.add_argument("--side", required=True, choices=["right", "left", "bimanual"])
  p.add_argument("--motion_file", required=True)
  p.add_argument("--checkpoint", required=True, help="Stage 1 base checkpoint.")
  p.add_argument("--obs_clip", type=float, default=5.0)
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
  args = p.parse_args()

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

  obs, _ = wrapped.reset()

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

  frames = []
  cmd = cast(ManipTransCommand, env.command_manager.get_term("motion"))
  motion_T = int(cmd.motion.time_step_totals.max().item())
  if args.max_steps <= 0:
    args.max_steps = max(1, motion_T - 1)
  if args.fps <= 0:
    args.fps = 1.0 / env.step_dt
  print(f"[rollout_base] motion_T={motion_T}  max_steps={args.max_steps}  fps={args.fps:.3f}")

  for step in range(args.max_steps):
    with torch.no_grad():
      actions = policy(obs)
    obs, _, _, _ = wrapped.step(actions)
    img = env.render()
    if img is not None:
      frames.append(img)

  env.close()

  out_path = Path(args.output)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimsave(str(out_path), frames, fps=args.fps)
  print(f"Saved {len(frames)} frames ({args.fps} fps) to {out_path}")


if __name__ == "__main__":
  main()
