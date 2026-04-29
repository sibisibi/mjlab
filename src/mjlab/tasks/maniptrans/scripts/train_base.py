"""Stage 1 base training — pure hand-only motion imitation.

The robot hand tracks the MANO reference in empty space. No object entity,
no pin, no contact sensors, no contact_match rewards, no pin_penalty,
no tactile obs, no object obs/rew/term.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import wandb

import mjlab.tasks.maniptrans.config  # noqa: F401  registers tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends


def build_env_cfg(args):
  task_id = f"mjlab-maniptrans-{args.robot}-{args.side}"
  cfg = load_env_cfg(task_id)
  cfg.scene.num_envs = args.num_envs
  cfg.commands["motion"].motion_file = args.motion_file
  return cfg


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--robot", required=True)
  p.add_argument("--side", required=True, choices=["right", "left", "bimanual"])
  p.add_argument("--input_dir", required=True)
  p.add_argument("--output_dir", required=True)
  p.add_argument("--index_path", required=True)
  p.add_argument("--indices", type=int, nargs="+", required=True)
  p.add_argument("--num_envs", type=int, required=True)
  p.add_argument("--max_iterations", type=int, default=1000000)
  p.add_argument("--save_interval", type=int, default=100)
  p.add_argument("--obs_clip", type=float, default=5.0)
  p.add_argument("--wandb_project", required=True)
  p.add_argument("--wandb_entity", required=True)
  p.add_argument("--wandb_tags", type=str, default="")
  p.add_argument("--group_name", required=True)
  p.add_argument("--exp_name", required=True)
  p.add_argument("--run_name", required=True)
  p.add_argument("--gpu", type=int, default=0)
  args = p.parse_args()

  with open(args.index_path) as f:
    rows = list(csv.DictReader(f))
  motion_filename = "motion.npz" if args.side == "bimanual" else f"motion_{args.side}.npz"
  motion_files = [
    f"{args.output_dir}/{args.robot}/{rows[i]['dataset']}/{rows[i]['filename']}/{motion_filename}"
    for i in args.indices
  ]
  args.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]

  configure_torch_backends()
  device = f"cuda:{args.gpu}"
  task_id = f"mjlab-maniptrans-{args.robot}-{args.side}"

  cfg = build_env_cfg(args)
  env = ManagerBasedRlEnv(cfg, device=device)

  agent_cfg = load_rl_cfg(task_id)
  agent_cfg.max_iterations = args.max_iterations
  agent_cfg.save_interval = args.save_interval
  agent_cfg.experiment_name = f"maniptrans_{args.robot}"
  agent_cfg.run_name = args.run_name
  agent_cfg.wandb_project = args.wandb_project
  agent_cfg.logger = "wandb"

  train_cfg = asdict(agent_cfg)
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

  runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
  env.close()


if __name__ == "__main__":
  main()
