"""Curriculum helpers for maniptrans residual training.

Mirrors ManipTrans (arXiv 2503.21860) curricula in mjlab's CurriculumManager idiom:

- Object-termination tightening uses the existing
  ``mjlab.envs.mdp.curriculums.termination_curriculum`` directly with stages
  built by :func:`build_obj_term_stages`. ManipTrans formula:
  ``scale = (e*2)^(-frac)*0.3 + 0.7`` with cubic dependence on pos/rot thresholds.
- Gravity ramp is :func:`gravity_curriculum` -- ``sim.model.opt.gravity`` is
  mutable at runtime in MuJoCo Warp, so we update it in-place each curriculum tick.

Step values are ``env.common_step_counter`` (single counter, ticks once per
``env.step()``). With ``num_steps_per_env=24`` (mjlab/rsl_rl default) one PPO
iter advances common_step_counter by 24.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def gravity_curriculum(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  schedule_steps: int = 1920,
  full_g: float = 9.81,
) -> dict[str, torch.Tensor]:
  """Linear ramp of -z gravity from 0 -> -full_g over schedule_steps env-steps.

  Mutates env.sim.model.opt.gravity[2] in place. Held at -full_g once
  schedule_steps is reached.

  Special case: ``schedule_steps <= 0`` skips the ramp entirely — gravity is
  -full_g from the very first step (constant mode).
  """
  s = int(env.common_step_counter)
  if schedule_steps <= 0:
    frac = 1.0
  else:
    frac = min(s / max(1, schedule_steps), 1.0)
  gz = -float(full_g) * frac
  env.sim.model.opt.gravity[2] = gz
  return {"gravity_z": torch.tensor(gz)}


def xfrc_curriculum(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  command_name: str = "motion",
  omega_n_start: float = 0.0,
  omega_n_end: float = 0.0,
  schedule_steps: int = 0,
  delay_steps: int = 0,
  zeta: float = 1.0,
  drive_omega_rot: bool = True,
) -> dict[str, torch.Tensor]:
  """Mass-normalized soft-attractor curriculum for `pin_mode="xfrc"`.

  Single knob: natural frequency ω_n (rad/s). Per-object mass is auto-detected
  from each object entity's body_mass. Gains derived as:

      kp_pos = m * ω_n^2
      kv_pos = 2 * ζ * sqrt(m * kp_pos) = 2 * ζ * m * ω_n

  Schedule: linear interpolation from ``omega_n_start`` → ``omega_n_end`` over
  ``schedule_steps`` env-steps. After that, held at omega_n_end. ``schedule_steps<=0``
  means held at omega_n_start from step 0 (constant mode — for the baseline).

  Mutates ``cmd.cfg.xfrc_kp_pos`` / ``xfrc_kv_pos`` each tick. Rotational gains
  are NOT touched (left at whatever cfg has — typically 0).

  Returns ``{xfrc_omega_n, xfrc_kp_pos, xfrc_kv_pos}`` for wandb logging
  (mirrors the gravity curriculum's logging pattern).
  """
  s = int(env.common_step_counter)
  s_post_delay = max(0, s - int(delay_steps))
  if schedule_steps <= 0:
    w = float(omega_n_start)
  else:
    frac = min(s_post_delay / max(1, schedule_steps), 1.0)
    w = float(omega_n_start) + frac * (float(omega_n_end) - float(omega_n_start))

  cmd = env.command_manager.get_term(command_name)
  obj_names = list((cmd.cfg.object_entity_names or {}).values())
  obj = env.scene[obj_names[0]]
  # body_mass is shape (B, nbody) batched per env. Index env 0 explicitly,
  # then sum over the entity's body_ids. Earlier bug: bare body_ids indexed
  # the BATCH dim instead, returning the entire env's mass (~10x too high).
  m = float(env.sim.model.body_mass[0, obj.indexing.body_ids].sum())

  kp = m * w * w
  kv = 2.0 * float(zeta) * m * w
  cmd.cfg.xfrc_kp_pos = kp
  cmd.cfg.xfrc_kv_pos = kv
  if drive_omega_rot:
    cmd.cfg.xfrc_omega_rot = float(w)
  return {
    "xfrc_omega_n": torch.tensor(w),
    "xfrc_kp_pos": torch.tensor(kp),
    "xfrc_kv_pos": torch.tensor(kv),
    "xfrc_obj_mass": torch.tensor(m),
    "xfrc_omega_rot": torch.tensor(float(w)) if drive_omega_rot else torch.tensor(0.0),
  }


def build_obj_term_stages(curriculum_scale: float = 1.0) -> tuple[list[dict], list[dict]]:
  """Build (pos_stages, rot_stages) for ManipTrans-style obj-termination tightening,
  scaled by ``curriculum_scale``.

  Returns lists of {"step": int, "params": {"threshold"|"threshold_deg": float}}
  suitable for ``mjlab.envs.mdp.curriculums.termination_curriculum``.

  Stages at fractions 0, 0.25, 0.5, 0.75, 1.0 of total span. Span =
  3200 frames * curriculum_scale (ManipTrans tighten_steps = 3200).
  """
  base_span = 3200
  span = int(base_span * curriculum_scale)
  pos_stages: list[dict] = []
  rot_stages: list[dict] = []
  for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
    scale = (math.e * 2.0) ** (-frac) * 0.3 + 0.7
    pos_thr = 0.02 / 0.343 * scale ** 3
    rot_thr = 30.0 / 0.343 * scale ** 3
    s = int(span * frac)
    pos_stages.append({"step": s, "params": {"threshold": float(pos_thr)}})
    rot_stages.append({"step": s, "params": {"threshold_deg": float(rot_thr)}})
  return pos_stages, rot_stages
