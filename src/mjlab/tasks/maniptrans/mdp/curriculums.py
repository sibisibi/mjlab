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
  """
  s = int(env.common_step_counter)
  frac = min(s / max(1, schedule_steps), 1.0)
  gz = -float(full_g) * frac
  env.sim.model.opt.gravity[2] = gz
  return {"gravity_z": torch.tensor(gz)}


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
