"""MLPModel subclass that clips normalized observations before the MLP.

rsl_rl's `EmpiricalNormalization.forward` does `(x - mean) / (std + eps)` with
no clipping. During normal single-policy training this is fine because PPO is
co-adapting the normalizer's running stats and the MLP weights on every
iteration — even rare out-of-distribution observations (e.g. contact-force
spikes at resets) get absorbed into the running stats transiently and the MLP
learns to handle them.

But if you train a base policy and then FREEZE it for use as a residual-policy
base (Stage 2 residual training), both the normalizer and the MLP are frozen.
Any OOD value that enters at inference time gets normalized to an extreme
z-score (sometimes 100+) and the frozen MLP extrapolates to absurd outputs,
blowing up the sim on the first step.

The standard mitigation used by rl_games, sb3's VecNormalize, RLLib, and
ManipTrans (which is why ManipTrans's frozen base works in Stage 2) is to
clip the normalized observations to `[-obs_clip, obs_clip]` before the MLP
sees them, BOTH during training and at inference. The MLP then learns on
bounded inputs and is intrinsically robust to OOD values post-freeze.

This class is a one-method override of `rsl_rl.models.MLPModel` that inserts
the clamp inside `get_latent()`, right after normalization. Everything else
(distribution, `forward`, weight init, update_normalization, etc.) is
inherited unchanged. The `obs_clip` kwarg is read from the model cfg dict.
"""

from __future__ import annotations

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from tensordict import TensorDict


class ClippedMLPModel(MLPModel):
  """MLPModel with obs clipping: `clamp(normalize(obs), -obs_clip, obs_clip)`."""

  def __init__(self, *args, obs_clip: float = 5.0, **kwargs) -> None:
    """Same signature as MLPModel plus a single extra kwarg.

    Args:
      obs_clip: Symmetric bound applied to normalized observations before the
        MLP. Typical values are 5.0 (rl_games, sb3) or 10.0 (more permissive).
        Must be > 0; a non-positive value is treated as a configuration error.
    """
    super().__init__(*args, **kwargs)
    if obs_clip <= 0:
      raise ValueError(
        f"ClippedMLPModel obs_clip must be > 0, got {obs_clip}. "
        f"Use the default MLPModel class if you don't want clipping."
      )
    self.obs_clip = float(obs_clip)

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    """Select obs groups, normalize, CLIP, return as MLP input latent."""
    obs_list = [obs[obs_group] for obs_group in self.obs_groups]
    latent = torch.cat(obs_list, dim=-1)
    latent = self.obs_normalizer(latent)
    latent = torch.clamp(latent, -self.obs_clip, self.obs_clip)
    return latent
