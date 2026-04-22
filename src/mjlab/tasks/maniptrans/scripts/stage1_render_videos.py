"""Render Stage 1 rollouts per trajectory with MANO ghost overlay.

For each trajectory in --render_indices (integer positions into the
preprocessed index.csv, same semantics as train_stage1_pinned.py), build a
1-env MuJoCo scene, run one deterministic rollout under the given
checkpoint, and save an mp4 with the MANO reference keypoints drawn as
semi-transparent spheres + bones on top of the robot.

Intended use: after training completes, render a curated subset of
trajectories to eyeball what the policy is doing. Typical call:

  python -m mjlab.tasks.maniptrans.scripts.stage1_render_videos \
      <all build_env_cfg flags matching training> \
      --checkpoint logs/.../model_1999.pt \
      --render_indices 948 1073 1172 1235 1382 1506 1571 1646 1709 1753 \
                       <plus 5 worst-performing from eval> \
      --output_dir logs/.../videos

The selection of the "5 worst" is up to the caller — the scorer logs
per-traj E_fingertip to wandb, pick from there.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import cast

import imageio
import numpy as np
import torch

import mjlab.tasks.maniptrans.config  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.maniptrans.mdp import ManipTransCommandCfg
from mjlab.tasks.maniptrans.mdp.commands import ManipTransCommand
from mjlab.tasks.maniptrans.scripts.train_stage1_pinned import build_env_cfg
from mjlab.tasks.registry import load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViewerConfig
from mjlab.viewer.debug_visualizer import DebugVisualizer


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
# Ghost color (light blue, 40% opacity).
GHOST_RGBA: tuple[float, float, float, float] = (0.4, 0.7, 1.0, 0.4)
JOINT_RADIUS = 0.008  # 8 mm spheres for keypoints
BONE_RADIUS = 0.003   # 3 mm capsules for bones
WRIST_RADIUS = 0.012  # slightly bigger for the wrist anchor


def _build_mano_bone_indices(joint_names: list[str]) -> list[tuple[int, int]]:
  """Return (parent_idx, child_idx) pairs for the MANO hand skeleton.

  Accepts either 3-level (proximal/intermediate/distal/tip) or 2-level
  (proximal/intermediate/tip) joint layouts — falls back by skipping
  missing names. Returns pairs of indices into joint_names; the caller
  also links wrist → each finger's proximal separately.
  """
  pairs: list[tuple[int, int]] = []
  level_names = ("proximal", "intermediate", "distal", "tip")
  for finger in FINGER_NAMES:
    chain_ids: list[int] = []
    for level in level_names:
      name = f"{finger}_{level}"
      if name in joint_names:
        chain_ids.append(joint_names.index(name))
    for a, b in zip(chain_ids[:-1], chain_ids[1:]):
      pairs.append((a, b))
  return pairs


def _make_mano_ghost_callback(
  env: ManagerBasedRlEnv,
  cmd: ManipTransCommand,
  orig_update_visualizers,
):
  """Return a new update_visualizers that also draws MANO ghosts for env 0."""
  side_list = list(cmd._side_list)
  # Pre-resolve bone indices per side (same joint_names in multi-traj runs).
  bones_per_side: dict[str, list[tuple[int, int]]] = {}
  prox_per_side: dict[str, list[int]] = {}
  for side in side_list:
    jn = cmd.motion.joint_names[side]
    bones_per_side[side] = _build_mano_bone_indices(jn)
    prox_per_side[side] = [
      jn.index(f"{f}_proximal") for f in FINGER_NAMES if f"{f}_proximal" in jn
    ]

  def _new_update_visualizers(visualizer: DebugVisualizer) -> None:
    orig_update_visualizers(visualizer)
    env_idx = 0  # We render env 0 only.
    # env_origins on device — move to CPU once for the call.
    origin = env.scene.env_origins[env_idx].detach().cpu().numpy()
    tr = int(cmd.env_traj_idx[env_idx].item())
    t = int(cmd.time_steps[env_idx].item())

    for side in side_list:
      wrist = cmd.motion.wrist_pos[side][tr, t].detach().cpu().numpy() + origin
      joints = cmd.motion.joints[side][tr, t].detach().cpu().numpy()  # (20, 3)
      joints_w = joints + origin  # (20, 3)

      # Spheres: wrist + 20 MANO joints.
      visualizer.add_sphere(wrist, WRIST_RADIUS, GHOST_RGBA)
      for j in joints_w:
        visualizer.add_sphere(j, JOINT_RADIUS, GHOST_RGBA)

      # Bones: wrist → each finger's proximal.
      for pidx in prox_per_side[side]:
        visualizer.add_cylinder(wrist, joints_w[pidx], BONE_RADIUS, GHOST_RGBA)
      # Finger-internal bones (proximal → intermediate → distal → tip).
      for a, b in bones_per_side[side]:
        visualizer.add_cylinder(joints_w[a], joints_w[b], BONE_RADIUS, GHOST_RGBA)

  return _new_update_visualizers


def render_one_trajectory(
  env: ManagerBasedRlEnv,
  wrapped: RslRlVecEnvWrapper,
  cmd: ManipTransCommand,
  policy,
  traj_local_idx: int,
  output_path: Path,
  device: str,
  fps: int = 60,
) -> dict[str, float]:
  """Reset env to traj_local_idx and render the full rollout to output_path.

  `traj_local_idx` is the index into the motion's trajectories (0-based,
  0..M-1), not the global index.csv index. For a single-traj run M=1 so
  traj_local_idx must be 0.

  Returns a dict with the rollout's mean E_fingertip_cm / E_wrist_pos_cm /
  E_wrist_rot_deg so the caller can cross-check against the wandb scorer.
  """
  # The env uses sampling_mode="start" with random traj selection on reset;
  # we force env 0 to traj_local_idx by overriding env_traj_idx after reset
  # and then stepping time back to 0 (reset already set time_steps=0 for
  # sampling_mode="start").
  _obs, _ = wrapped.reset()
  cmd.env_traj_idx[0] = traj_local_idx
  cmd.time_steps[0] = 0
  # Re-do the motion's per-reset warm-start with the new traj assignment so
  # the initial robot state matches the chosen trajectory's frame 0, not the
  # one randomly drawn by reset().
  cmd._resample_command(torch.tensor([0], device=device, dtype=torch.long))
  # Let obs recompute next step; on the first policy call obs is still the
  # pre-override one, which is fine — the tracking just starts one frame late.

  T_m = int(cmd.motion.time_step_totals[traj_local_idx].item())
  frames: list[np.ndarray] = []
  sum_tip = 0.0
  sum_wrist_pos = 0.0
  sum_wrist_rot = 0.0
  count = 0

  # First render frame (pre-step) so mp4 starts from the reset pose.
  img = env.render()
  if img is not None:
    frames.append(img)

  obs = wrapped.get_observations().to(device)
  with torch.no_grad():
    for _step in range(T_m):
      actions = policy(obs)
      obs, _r, _d, _e = wrapped.step(actions)
      img = env.render()
      if img is not None:
        frames.append(img)
      # Metric trace on env 0 (side-mean).
      tip = torch.norm(
        cmd.mano_tip_pos_w[0] - cmd.robot_tip_pos_w[0], dim=-1
      ).mean().item()
      wpos = torch.norm(
        cmd.mano_wrist_pos_w[0] - cmd.robot_wrist_pos_w[0], dim=-1
      ).mean().item()
      R_diff = cmd.mano_wrist_rot_w[0] @ cmd.robot_wrist_rot_w[0].transpose(-1, -2)
      trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
      cos_t = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
      wrot_deg = (torch.acos(cos_t) * (180.0 / np.pi)).mean().item()
      sum_tip += tip
      sum_wrist_pos += wpos
      sum_wrist_rot += wrot_deg
      count += 1

  output_path.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimsave(str(output_path), frames, fps=fps)

  inv = 1.0 / max(count, 1)
  return {
    "tip_pos_cm": sum_tip * inv * 100.0,
    "wrist_pos_cm": sum_wrist_pos * inv * 100.0,
    "wrist_rot_deg": sum_wrist_rot * inv,
    "n_frames": len(frames),
  }


def main() -> None:
  p = argparse.ArgumentParser()
  # --- Full build_env_cfg flag set (must match training) ---
  p.add_argument("--robot", required=True)
  p.add_argument("--side", required=True)
  p.add_argument("--index_path", required=True)
  p.add_argument("--indices", type=int, nargs="+", required=True,
    help="Indices used at training time (needed to rebuild the exact "
         "multi-traj env with the same object validation). This scorer will "
         "reset env 0 to each of --render_indices in turn.")
  p.add_argument("--output_dir", required=True,
    help="Preprocessing output root (same as training's --output_dir).")
  p.add_argument("--obj_density", type=float, required=True)
  p.add_argument("--wrist_residual_scale", type=float, required=True)
  p.add_argument("--finger_residual_scale", type=float, required=True)
  p.add_argument("--contact_match_weight", type=float, required=True)
  p.add_argument("--contact_match_beta", type=float, required=True)
  p.add_argument("--contact_match_A", type=float, required=True)
  p.add_argument("--contact_match_eps", type=float, required=True)
  p.add_argument("--pin_mode", choices=("hard", "actuated", "xfrc"), default="hard")
  p.add_argument("--pin_interval", type=int, default=6)
  p.add_argument("--object_kp_pos", type=float, default=0.0)
  p.add_argument("--object_kv_pos", type=float, default=0.0)
  p.add_argument("--object_kp_rot", type=float, default=0.0)
  p.add_argument("--object_kv_rot", type=float, default=0.0)
  p.add_argument("--xfrc_kp_pos", type=float, default=0.0)
  p.add_argument("--xfrc_kv_pos", type=float, default=0.0)
  p.add_argument("--xfrc_kp_rot", type=float, default=0.0)
  p.add_argument("--xfrc_kv_rot", type=float, default=0.0)
  p.add_argument("--enable_tactile", action="store_true")
  p.add_argument("--no_contact_missing", action="store_true")
  p.add_argument("--enable_object_term", action="store_true")
  p.add_argument("--enable_object_obs_actor", action="store_true")
  p.add_argument("--enable_object_obs_critic", action="store_true")
  p.add_argument("--enable_object_rew", action="store_true")
  p.add_argument("--object_reward_mult", type=float, default=1.0)
  p.add_argument("--actor_no_hand_obj_distance", action="store_true")
  p.add_argument("--actor_no_gt_tips_distance", action="store_true")
  p.add_argument("--obs_clip", type=float, default=5.0)
  p.add_argument("--ccd_iterations", type=int, default=200)
  p.add_argument("--no_object", action="store_true",
    help="Must match training if training was done with --no_object.")
  # --- Scorer / render-specific ---
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--render_indices", type=int, nargs="+", required=True,
    help="Subset of global index.csv indices to render. Each must also be "
         "present in --indices (the training set). One mp4 per index.")
  p.add_argument("--output_videos_dir", required=True,
    help="Where to save the mp4s. Filename pattern: idx{i}.mp4")
  p.add_argument("--fps", type=int, default=60)
  p.add_argument("--azimuth", type=float, default=90.0)
  p.add_argument("--elevation", type=float, default=-25.0)
  p.add_argument("--distance", type=float, default=0.8)
  p.add_argument("--seed", type=int, default=0)
  # --- Unused-but-required by build_env_cfg signature ---
  p.add_argument("--num_envs", type=int, default=1)
  p.add_argument("--max_iterations", type=int, default=1)
  p.add_argument("--save_interval", type=int, default=1)
  args = p.parse_args()

  configure_torch_backends()
  torch.manual_seed(args.seed)
  device = "cuda:0"
  task_id = f"mjlab-maniptrans-{args.robot}-{args.side}"

  # Resolve motion files from index.csv (same as train_stage1_pinned.py)
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
  args.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]
  args.data_dir = data_dirs[0]
  args._all_data_dirs = data_dirs

  # Map requested render_indices → local traj positions (0..M-1).
  idx_to_local: dict[int, int] = {gi: li for li, gi in enumerate(args.indices)}
  render_local: list[tuple[int, int]] = []  # (global_idx, local_idx)
  for gi in args.render_indices:
    if gi not in idx_to_local:
      raise ValueError(
        f"--render_indices contains {gi} which is not in --indices "
        f"(training set). Every video-rendered index must be part of the "
        f"multi-traj env."
      )
    render_local.append((gi, idx_to_local[gi]))

  # Build env: num_envs=1, render_mode="rgb_array", sampling_mode="start",
  # terminations cleared, obs corruption off.
  args.num_envs = 1
  cfg = build_env_cfg(args)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, ManipTransCommandCfg)
  motion_cmd.sampling_mode = "start"
  cfg.scene.num_envs = 1
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
  cmd = cast(ManipTransCommand, env.command_manager.get_term("motion"))

  # Install MANO ghost overlay.
  orig_upd = env.update_visualizers
  env.update_visualizers = _make_mano_ghost_callback(env, cmd, orig_upd)

  # --- Load checkpoint ---
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

  # --- Render each selected trajectory ---
  out_dir = Path(args.output_videos_dir)
  summary: list[tuple[int, dict[str, float]]] = []
  for global_idx, local_idx in render_local:
    out_path = out_dir / f"idx{global_idx}.mp4"
    print(f"[render] idx={global_idx} (local={local_idx}) → {out_path}")
    metrics = render_one_trajectory(
      env, wrapped, cmd, policy, local_idx, out_path, device, fps=args.fps,
    )
    summary.append((global_idx, metrics))
    print(
      f"  frames={metrics['n_frames']}  "
      f"E_fingertip={metrics['tip_pos_cm']:.2f}cm  "
      f"E_wrist_pos={metrics['wrist_pos_cm']:.2f}cm  "
      f"E_wrist_rot={metrics['wrist_rot_deg']:.2f}deg"
    )

  env.close()

  print("=" * 72)
  print(f"Rendered {len(summary)} trajectory videos to {out_dir}")
  for gi, m in summary:
    print(f"  idx{gi:>5d}  tip={m['tip_pos_cm']:5.2f}cm  "
          f"wrist_pos={m['wrist_pos_cm']:5.2f}cm  "
          f"wrist_rot={m['wrist_rot_deg']:5.2f}deg")


if __name__ == "__main__":
  main()
