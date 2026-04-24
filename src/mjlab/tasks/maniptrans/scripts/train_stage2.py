"""Train Stage 2 residual policy for ManipTrans.

Loads a frozen Stage 1 base checkpoint, builds a residual MLP on top, and
trains only the residual head while the base stays frozen. Shares the env
builder with `train_stage1_pinned.py` via `build_env_cfg(args)` so the
residual env's actor observations start with the base's training obs as a
contiguous prefix — which is required for the `obs[:base_obs_dim]` slice
inside `rl/residual_actor.py::ResidualActor.get_latent`.

Stage 2 differs from Stage 1 in two places:

1. `motion_cmd.pin_objects = False` — objects are free (no pinning).
2. `train_cfg["actor"]["residual_action_scale"] = args.residual_action_scale`
   — the residual actor composes `applied = base + residual * scale` inside
   its `forward()`. The action term's `action_dim` stays at `n_dofs`; there
   is no 2x doubling or split.

The residual actor is swapped in by rewriting the `actor` section of the
rsl_rl train_cfg to point to `mjlab.tasks.maniptrans.rl.residual_actor.ResidualActor`,
passing `base_checkpoint`, `base_obs_dim`, and `base_action_dim` so the
frozen base loads its own weights at construction time.

Usage:
  python -m mjlab.tasks.maniptrans.scripts.train_stage2 \
      --robot xhand --side bimanual \
      --motion_file data/xhand/oakink2/.../0/motion.npz \
      --data_dir data/oakink2/.../0 \
      --base_checkpoint logs/rsl_rl/.../model_1400.pt \
      --residual_action_scale 1.0 \
      [same Stage 1 flags matching the base's training: --enable_tactile,
       --actor_no_hand_obj_distance, --actor_no_gt_tips_distance,
       --enable_object_obs_critic, --enable_object_rew --object_reward_mult 1.0,
       --contact_match_weight 1.0 --contact_match_beta 40 --contact_match_A 1.0
       --contact_match_eps 0.5, etc.]
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import wandb

import mjlab.tasks.maniptrans.config  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.maniptrans.mdp import ManipTransCommandCfg
from mjlab.tasks.maniptrans.scripts.train_stage1_pinned import build_env_cfg
from mjlab.tasks.registry import load_rl_cfg
from mjlab.utils.torch import configure_torch_backends


def main():
  p = argparse.ArgumentParser()
  # --- Shared args with train_stage1_pinned.py (same semantics) ---
  p.add_argument("--robot", required=True)
  p.add_argument("--side", required=True)
  p.add_argument("--index_path", required=True,
    help="Path to the dataset index CSV.")
  p.add_argument("--indices", type=int, nargs="+", required=True,
    help="One or more integer indices into the index CSV.")
  p.add_argument("--output_dir", required=True,
    help="Preprocessing output directory.")
  p.add_argument("--obj_density", type=float, required=True)
  p.add_argument("--wrist_residual_scale", type=float, required=True)
  p.add_argument("--finger_residual_scale", type=float, required=True)
  p.add_argument("--contact_match_weight", type=float, required=True)
  p.add_argument("--contact_match_beta", type=float, required=True)
  p.add_argument("--contact_match_gamma", type=float, default=200.0)
  p.add_argument("--contact_match_tol", type=float, default=0.002)
  p.add_argument("--contact_match_force_cap", type=float, default=30.0)
  p.add_argument("--adaptive_pin", action="store_true")
  p.add_argument("--pin_pos_threshold", type=float, default=0.030)
  p.add_argument("--pin_rot_threshold", type=float, default=1.5708)
  p.add_argument("--pin_penalty_weight", type=float, default=0.0)
  # Pin flags still required by build_env_cfg; for Stage 2 we override
  # pin_objects=False after the call, so --pin_mode and --pin_interval are
  # effectively ignored but must be passed to satisfy build_env_cfg's args.
  p.add_argument("--pin_mode", choices=("hard", "actuated", "xfrc"), default="hard")
  p.add_argument("--pin_interval", type=int, default=1)
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
  # --- Stage 2 specific args ---
  p.add_argument("--base_checkpoint", required=True,
    help="Path to frozen Stage 1 checkpoint. Its actor_state_dict['obs_normalizer._mean'] "
         "dimension defines base_obs_dim; its last mlp layer bias dimension defines "
         "base_action_dim.")
  p.add_argument("--residual_action_scale", type=float, required=True,
    help="Residual action scale > 0 enables Stage 2 action splitting in ManipTransAction. "
         "Raw policy output is 2*n_dofs; applied action = first_half + second_half * scale. "
         "ManipTrans default is 2.0.")
  p.add_argument("--init_std", type=float, default=0.37,
    help="Initial std for the residual GaussianDistribution. ManipTrans default is 0.37 "
         "(= exp(-1), from const_initializer val=-1 on log_std).")
  p.add_argument("--lr", type=float, default=2.0e-4,
    help="PPO learning rate. ManipTrans Stage 2 default is 2e-4.")
  p.add_argument("--value_loss_coef", type=float, default=4.0,
    help="PPO value loss coefficient. ManipTrans Stage 2 default is 4.0.")
  p.add_argument("--obs_clip", type=float, default=0.0,
    help="Clamp normalized obs to [-obs_clip, obs_clip] before the MLP, "
         "symmetric with the base's Stage 1 ClippedMLPModel. Applied to both "
         "the ResidualActor (trainable residual path) and the critic. The "
         "frozen base inside ResidualActor has its own internal clip at 5.0 "
         "regardless of this flag. Default 0.0 = disabled.")
  p.add_argument("--ccd_iterations", type=int, default=50,
    help="MuJoCo CCD iteration cap passed through to build_env_cfg. Match the "
         "base's training value (bases trained 2026-04-14 onward use 200).")
  # --- Training / logging ---
  p.add_argument("--num_envs", type=int, required=True)
  p.add_argument("--max_iterations", type=int, required=True)
  p.add_argument("--save_interval", type=int, required=True)
  p.add_argument("--wandb_project", required=True)
  p.add_argument("--wandb_entity", required=True,
    help="wandb entity (team/user). No default — set explicitly per run.")
  p.add_argument("--wandb_tags", type=str, default="")
  p.add_argument("--group_name", required=True,
    help="High-level ablation study this run belongs to (maps to wandb `group`).")
  p.add_argument("--exp_name", required=True,
    help="Specific ablation variant within the group (maps to wandb `job_type`).")
  p.add_argument("--run_name", required=True)
  p.add_argument("--gpu", type=int, required=True)
  p.add_argument("--eval_interval", type=int, default=0,
    help="Run eval every N iters and log eval/SR_at_* + eval/E_* to wandb. "
         "Also runs at the final iteration. 0 = disabled (default).")
  p.add_argument("--eval_rollouts", type=int, default=100,
    help="Number of parallel rollouts per eval. 100 gives ~±5 pp CI on SR.")
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
  args.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]
  args.data_dir = data_dirs[0]
  args._all_data_dirs = data_dirs  # consumed by build_env_cfg for multi-traj object validation

  configure_torch_backends()
  device = f"cuda:{args.gpu}"
  task_id = f"mjlab-maniptrans-{args.robot}-{args.side}"

  # Build env cfg using the same helper as Stage 1. The residual env's actor
  # obs is a SUPERSET of the base's actor obs with new terms appended at the
  # end (dict insertion order), so the first `base_obs_dim` dims of the
  # residual's full obs equal the base's training obs.
  cfg = build_env_cfg(args)

  # --- Stage 2 overrides ---
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, ManipTransCommandCfg)
  motion_cmd.pin_objects = False  # free objects — residual trains on unpinned sim

  # NOTE: `residual_action_scale` is a ResidualActor kwarg (NOT an action term
  # field). The action term keeps `action_dim = n_dofs` and the residual
  # composition `applied = base + residual * scale` happens inside the actor.

  # --- Load base checkpoint dims (for ResidualActor init) ---
  ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
  ckpt_sd = ckpt["actor_state_dict"]
  base_obs_dim = int(ckpt_sd["obs_normalizer._mean"].shape[-1])
  # Last mlp layer bias shape = base's action_dim (= n_dofs for Stage 1).
  last_mlp_idx = max(int(k.split(".")[1]) for k in ckpt_sd if k.startswith("mlp."))
  base_action_dim = int(ckpt_sd[f"mlp.{last_mlp_idx}.bias"].shape[0])
  del ckpt, ckpt_sd

  # --- Create env ---
  env = ManagerBasedRlEnv(cfg, device=device)

  # --- Eval env (separate instance with free-object Stage 2 semantics,
  # smaller num_envs, play-mode sampling, all terminations stripped so
  # every rollout reaches motion end). Rebuilt via build_env_cfg with
  # overridden args.num_envs so it inherits physics/obs/rewards exactly. ---
  eval_env = None
  eval_wrapped = None
  if args.eval_interval > 0:
    _orig_num_envs = args.num_envs
    args.num_envs = args.eval_rollouts
    try:
      eval_cfg = build_env_cfg(args)
    finally:
      args.num_envs = _orig_num_envs

    eval_motion_cmd = eval_cfg.commands["motion"]
    assert isinstance(eval_motion_cmd, ManipTransCommandCfg)
    eval_motion_cmd.pin_objects = False  # match Stage 2 training
    eval_motion_cmd.sampling_mode = "start"
    # Clear ALL terminations so eval rollouts always reach T_m; SR's
    # time-averaged check runs over the full [15, T-15) window.
    eval_cfg.terminations = {}
    for _group_name in ("actor", "critic"):
      if _group_name in eval_cfg.observations:
        eval_cfg.observations[_group_name].enable_corruption = False

    eval_env = ManagerBasedRlEnv(eval_cfg, device=device)
    eval_wrapped = RslRlVecEnvWrapper(eval_env)

  # --- RL config ---
  agent_cfg = load_rl_cfg(task_id)
  agent_cfg.max_iterations = args.max_iterations
  agent_cfg.save_interval = args.save_interval
  agent_cfg.algorithm.learning_rate = args.lr
  agent_cfg.algorithm.schedule = "fixed"
  agent_cfg.algorithm.value_loss_coef = args.value_loss_coef
  agent_cfg.experiment_name = f"maniptrans_{args.robot}_stage2"
  agent_cfg.run_name = args.run_name
  agent_cfg.wandb_project = args.wandb_project
  agent_cfg.logger = "wandb"

  # Override actor to the residual class
  agent_cfg.actor.hidden_dims = (256, 512, 128, 64)
  agent_cfg.actor.activation = "elu"
  agent_cfg.actor.obs_normalization = True
  agent_cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": args.init_std,
    "std_type": "scalar",
  }

  train_cfg = asdict(agent_cfg)
  train_cfg["actor"]["class_name"] = (
    "mjlab.tasks.maniptrans.rl.residual_actor.ResidualActor"
  )
  train_cfg["actor"]["base_checkpoint"] = args.base_checkpoint
  train_cfg["actor"]["base_obs_dim"] = base_obs_dim
  train_cfg["actor"]["base_action_dim"] = base_action_dim
  train_cfg["actor"]["residual_action_scale"] = args.residual_action_scale

  # Symmetric obs clipping with the Stage 1 base. When --obs_clip > 0:
  # - ResidualActor clamps its own normalized obs in get_latent (obs_clip kwarg).
  # - Critic is swapped to ClippedMLPModel, matching the base's Stage 1 critic.
  # The frozen base's internal clip is independent and always active at 5.0.
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

  # --- Periodic eval hook via logger monkey-patch (mirrors Stage 1 at
  # train_stage1_pinned.py:707–739). Fires evaluate_stage2 +
  # log_stage2_eval_to_wandb at every eval_interval completed iters and
  # at the final iter. ---
  if eval_wrapped is not None and args.eval_interval > 0:
    from mjlab.tasks.maniptrans.scripts.stage2_scorer import (
      evaluate_stage2, log_stage2_eval_to_wandb,
    )

    _orig_log = runner.logger.log
    _max_it = args.max_iterations
    _eval_iv = args.eval_interval

    def _log_with_eval(**kwargs):
      _orig_log(**kwargs)
      it = kwargs.get("it")
      if it is None:
        return
      completed = it + 1
      fire = (completed % _eval_iv == 0) or (completed == _max_it)
      if not fire:
        return
      policy = runner.get_inference_policy(device=device)
      metrics = evaluate_stage2(eval_env, eval_wrapped, policy, device)
      log_stage2_eval_to_wandb(metrics, iter_idx=completed)
      g = metrics["global"]
      print(
        f"[eval @ iter {completed}/{_max_it}]  "
        f"SR@1.0={g['SR_at_1p0'] * 100:5.1f}%  "
        f"E_obj_pos={g['E_obj_pos_cm']:6.2f} cm  "
        f"E_obj_rot={g['E_obj_rot_deg']:6.2f} deg  "
        f"E_ft={g['E_fingertip_cm']:5.2f} cm  "
        f"(n_envs={g['n_valid_envs']}/{g['n_envs']})"
      )

    runner.logger.log = _log_with_eval

  runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
  env.close()
  if eval_env is not None:
    eval_env.close()


if __name__ == "__main__":
  main()
