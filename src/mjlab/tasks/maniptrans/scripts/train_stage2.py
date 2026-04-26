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
      --input_dir $DATA_DIR \
      --output_dir $DATA_DIR/handphuma \
      --index_path $DATA_DIR/oakink2/index.csv \
      --indices 1854 \
      --base_checkpoint logs/rsl_rl/.../model_1400.pt \
      --residual_action_scale 1.0 \
      [same Stage 1 flags matching the base's training: --enable_tactile,
       --actor_no_hand_obj_distance, --actor_no_gt_tips_distance,
       --enable_object_obs_critic, --enable_object_rew --object_reward_mult 1.0,
       --contact_match_weight 1.0 --contact_match_beta 40, etc.]
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
  p.add_argument("--side", required=True, choices=["right", "left", "bimanual"])
  p.add_argument("--index_path", required=True,
    help="Path to the dataset index CSV.")
  p.add_argument("--indices", type=int, nargs="+", required=True,
    help="One or more integer indices into the index CSV.")
  p.add_argument("--input_dir", required=True,
    help="Pool root for object meshes. Resolves to "
         "<input_dir>/<dataset>/objects/<obj_id>/{visual.obj, convex/}.")
  p.add_argument("--output_dir", required=True,
    help="Preprocessing output directory. Motion at "
         "<output_dir>/<robot>/<dataset>/<filename>/motion.npz, task_info at "
         "<output_dir>/<dataset>/<filename>/task_info.json.")
  p.add_argument("--obj_density", type=float, default=800.0)
  p.add_argument("--contact_match_weight", type=float, required=True)
  p.add_argument("--contact_match_beta", type=float, default=40.0)
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
  base_group = p.add_mutually_exclusive_group(required=True)
  base_group.add_argument("--base_checkpoint",
    help="Path to a single frozen Stage 1 checkpoint (single-hand residual mode, "
         "--side {right,left}). Its actor_state_dict['obs_normalizer._mean'] "
         "dimension defines base_obs_dim; its last mlp layer bias dimension defines "
         "base_action_dim.")
  base_group.add_argument("--base_checkpoints", nargs=2, metavar=("RIGHT", "LEFT"),
    help="Two per-side frozen Stage 1 checkpoints (bimanual residual mode, "
         "--side bimanual). Order is [right, left]. Both ckpts must share "
         "base_obs_dim and base_action_dim.")
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
  p.add_argument("--obs_clip", type=float, default=5.0,
    help="Clamp normalized obs to [-obs_clip, obs_clip] before the MLP, "
         "symmetric with the base's Stage 1 ClippedMLPModel. Applied to both "
         "the ResidualActor (trainable residual path) and the critic. The "
         "frozen base inside ResidualActor has its own internal clip at 5.0 "
         "regardless of this flag. 0.0 disables clipping.")
  # --- Training / logging ---
  p.add_argument("--num_envs", type=int, required=True)
  p.add_argument("--max_iterations", type=int, default=1000000)
  p.add_argument("--save_interval", type=int, default=100)
  p.add_argument("--wandb_project", required=True)
  p.add_argument("--wandb_entity", required=True,
    help="wandb entity (team/user). No default — set explicitly per run.")
  p.add_argument("--wandb_tags", type=str, default="")
  p.add_argument("--group_name", required=True,
    help="High-level ablation study this run belongs to (maps to wandb `group`).")
  p.add_argument("--exp_name", required=True,
    help="Specific ablation variant within the group (maps to wandb `job_type`).")
  p.add_argument("--run_name", required=True)
  p.add_argument("--gpu", type=int, default=0)
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
  args.pool_dir = f"{args.input_dir}/{rows[args.indices[0]]['dataset']}"

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

  # --- Resolve base checkpoint(s) and validate against --side ---
  if args.base_checkpoints is not None:
    base_ckpts = list(args.base_checkpoints)
    if args.side != "bimanual":
      raise ValueError(
        f"--base_checkpoints (2 paths) requires --side bimanual; got --side {args.side}."
      )
  else:
    base_ckpts = [args.base_checkpoint]
    if args.side not in ("right", "left"):
      raise ValueError(
        f"--base_checkpoint (1 path) requires --side {{right,left}}; got --side {args.side}."
      )

  def _read_base_dims(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["actor_state_dict"]
    obs_dim = int(sd["obs_normalizer._mean"].shape[-1])
    last_idx = max(int(k.split(".")[1]) for k in sd if k.startswith("mlp."))
    act_dim = int(sd[f"mlp.{last_idx}.bias"].shape[0])
    return obs_dim, act_dim

  base_dims = [_read_base_dims(p) for p in base_ckpts]
  if len(base_dims) == 2 and base_dims[0] != base_dims[1]:
    raise ValueError(
      f"Per-side bases disagree on dims: right={base_dims[0]} vs left={base_dims[1]}."
    )
  base_obs_dim, base_action_dim = base_dims[0]

  # --- Create env ---
  env = ManagerBasedRlEnv(cfg, device=device)

  # --- Bimanual two-base mode: extract per-side term specs + n_wrist_per_side ---
  # Most hand-obs terms are per-side concat (bimanual dim = 2 * per-side dim).
  # `wrist_state` is the lone exception: it reads `hand.data.root_link_*` which
  # is one tensor for the whole bimanual entity (13D), so its bimanual dim
  # equals its per-side dim. Both bases see the same shared wrist_state tensor.
  GLOBAL_HAND_OBS_TERMS = {"wrist_state"}
  perside_term_specs: list[tuple[int, bool]] | None = None
  n_wrist_per_side = 0
  if len(base_ckpts) == 2:
    active = env.observation_manager.active_terms["actor"]
    shapes = env.observation_manager.group_obs_term_dim["actor"]
    perside_term_specs = []
    accum = 0
    bimanual_prefix_sum = 0
    for name, shape in zip(active, shapes):
      bim_dim = int(shape[-1]) if isinstance(shape, (tuple, list)) else int(shape)
      is_global = name in GLOBAL_HAND_OBS_TERMS
      if is_global:
        per_side_dim = bim_dim
      else:
        if bim_dim % 2 != 0:
          raise RuntimeError(
            f"Per-side hand obs term {name!r} has odd bimanual dim {bim_dim}; "
            f"either it should be in GLOBAL_HAND_OBS_TERMS or bimanual layout is broken."
          )
        per_side_dim = bim_dim // 2
      perside_term_specs.append((per_side_dim, is_global))
      accum += per_side_dim
      bimanual_prefix_sum += bim_dim
      if accum == base_obs_dim:
        break
      if accum > base_obs_dim:
        raise RuntimeError(
          f"Per-side hand obs accum {accum} overshot base_obs_dim {base_obs_dim} "
          f"at term {name!r}; check GLOBAL_HAND_OBS_TERMS membership."
        )
    if accum != base_obs_dim:
      raise RuntimeError(
        f"Per-side hand obs prefix sum {accum} != base_obs_dim {base_obs_dim}. "
        f"Specs: {perside_term_specs}."
      )
    bimanual_n_wrist = env.action_manager.get_term("maniptrans")._n_wrist
    if bimanual_n_wrist % 2 != 0:
      raise RuntimeError(
        f"Bimanual n_wrist={bimanual_n_wrist} is not even; cannot split per side."
      )
    n_wrist_per_side = bimanual_n_wrist // 2
    env_action_dim = env.action_manager.total_action_dim
    if env_action_dim != 2 * base_action_dim:
      raise RuntimeError(
        f"Bimanual env action_dim ({env_action_dim}) != 2 * base_action_dim "
        f"({2 * base_action_dim})."
      )
    print(f"[stage2] bimanual: base_obs_dim={base_obs_dim} base_action_dim={base_action_dim} "
          f"n_wrist_per_side={n_wrist_per_side} env_action_dim={env_action_dim}")
    print(f"[stage2] bimanual_hand_obs_dim={bimanual_prefix_sum} (sum of bimanual prefix dims)")
    print(f"[stage2] perside_term_specs (sum={accum}): {perside_term_specs}")

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
  train_cfg["actor"]["base_checkpoints"] = base_ckpts
  train_cfg["actor"]["base_obs_dim"] = base_obs_dim
  train_cfg["actor"]["base_action_dim"] = base_action_dim
  train_cfg["actor"]["residual_action_scale"] = args.residual_action_scale
  if perside_term_specs is not None:
    train_cfg["actor"]["perside_term_specs"] = perside_term_specs
    train_cfg["actor"]["n_wrist_per_side"] = n_wrist_per_side

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

  runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
  env.close()


if __name__ == "__main__":
  main()
