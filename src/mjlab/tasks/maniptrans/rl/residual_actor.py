"""Residual actor for Stage 2 training.

Classical residual policy: `applied_action = base_action + residual * scale`.

- Frozen base MLP (loaded from Stage 1 checkpoint). Output 36 dims (n_dofs).
- Trainable residual MLP. Output 36 dims (same as base).
- Applied action fed to the env: `base_action + residual_mlp_out * residual_action_scale`.
- `GaussianDistribution` is centered at `applied_mean`, with its own learnable std.
- `residual_mlp` last layer initialized near zero (xavier_normal gain=0.01), so at
  iter 0 the applied action ≈ base action → the hand starts behaving like Stage 1.

Note the action term (`ManipTransAction`) does NOT double its action_dim in Stage
2 — the residual composition happens here, inside the actor. The env still sees
a single `n_dofs`-dim action.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules import MLP, EmpiricalNormalization, GaussianDistribution
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict


class FrozenBaseModel(nn.Module):
  """Frozen Stage 1 base model. Forward returns the MLP's deterministic mean.

  - Base contributes its DETERMINISTIC mean action (no stochastic sampling).
    Exploration is the residual actor's responsibility via its own distribution.
  - Normalized obs are CLIPPED to `[-obs_clip, obs_clip]` before the MLP sees
    them. The base's normalizer was trained via a running average during Stage
    1 and converged to very narrow stats for dims with rare-but-large spikes
    (e.g. contact forces at reset). In Stage 2 the frozen normalizer can't
    widen, so unclipped OOD values produce z-scores in the hundreds, and the
    frozen MLP extrapolates to absurd outputs. Clipping caps the z-score so the
    frozen MLP always sees inputs within its training range.
  """

  def __init__(self, checkpoint_path: str, obs_dim: int, action_dim: int,
               hidden_dims: tuple[int, ...], activation: str,
               obs_clip: float = 5.0):
    super().__init__()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor_sd = ckpt["actor_state_dict"]

    self.obs_normalizer = EmpiricalNormalization(obs_dim)
    self.mlp = MLP(obs_dim, action_dim, hidden_dims, activation)
    self.obs_clip = float(obs_clip)

    own_sd = {k: v for k, v in actor_sd.items()
              if k.startswith("obs_normalizer.") or k.startswith("mlp.")}
    self.load_state_dict(own_sd, strict=False)

    for param in self.parameters():
      param.requires_grad = False

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    normalized = self.obs_normalizer(obs)
    normalized = torch.clamp(normalized, -self.obs_clip, self.obs_clip)
    return self.mlp(normalized)

  def train(self, mode: bool = True):
    super().train(False)
    return self


class ResidualActor(nn.Module):
  """Residual actor: frozen stochastic base + trainable residual MLP.

  Architecture:
    1. Frozen base: obs[:base_obs_dim] → sample from base dist → base_action
    2. Normalize full obs → cat(obs_norm, base_action) → residual MLP → 2x action_dim
    3. Distribution wraps full 2x output
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int],
    activation: str,
    obs_normalization: bool,
    distribution_cfg: dict | None,
    base_checkpoint: str,
    base_obs_dim: int,
    base_action_dim: int,
    residual_action_scale: float,
    obs_clip: float = 0.0,
  ):
    super().__init__()

    self.obs_groups = obs_groups.get(obs_set, [obs_set])
    obs_list = [obs[g] for g in self.obs_groups]
    self.obs_dim = sum(o.shape[-1] for o in obs_list)

    self.base_obs_dim = base_obs_dim
    self.base_action_dim = base_action_dim
    self._residual_action_scale = float(residual_action_scale)
    self.obs_clip = float(obs_clip)
    # output_dim must equal n_dofs (the env's action_dim). Sanity check — if
    # the action term is still doubling we'd hit a shape mismatch at forward.
    assert output_dim == base_action_dim, (
      f"ResidualActor expects output_dim ({output_dim}) == base_action_dim "
      f"({base_action_dim}); the action term must not double action_dim in "
      f"Stage 2 (residual composition is handled by this class)."
    )

    # Frozen base model (loaded from Stage 1 checkpoint). Its MLP outputs
    # `base_action_dim` = n_dofs. Its distribution produces stochastic samples
    # so the residual has a proper exploration anchor at init.
    self.base = FrozenBaseModel(
      base_checkpoint, base_obs_dim, base_action_dim, hidden_dims, activation,
    )

    # Obs normalizer
    self.obs_normalization = obs_normalization
    if obs_normalization:
      self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
    else:
      self.obs_normalizer = nn.Identity()

    # Distribution over the APPLIED action (n_dofs). Its mean is set to
    # `base_action + residual * scale` each forward(), while std is a learnable
    # parameter of the distribution itself.
    if distribution_cfg is not None:
      dist_cfg = dict(distribution_cfg)
      dist_class_name = dist_cfg.pop("class_name", "GaussianDistribution")
      if dist_class_name == "GaussianDistribution":
        self.distribution = GaussianDistribution(output_dim, **dist_cfg)
      else:
        dist_class = resolve_callable(dist_class_name)
        self.distribution = dist_class(output_dim, **dist_cfg)
      mlp_output_dim = self.distribution.input_dim
    else:
      self.distribution = None
      mlp_output_dim = output_dim

    # Residual MLP: input = obs_norm + base_action, output = n_dofs (the
    # residual, added on top of the frozen base sample via residual_action_scale).
    residual_input_dim = self.obs_dim + base_action_dim
    self.residual_mlp = MLP(residual_input_dim, mlp_output_dim, hidden_dims, activation)

    # Init residual output near zero (ManipTrans: glorot_normal gain=0.01)
    last_layer = list(self.residual_mlp.modules())[-1]
    if isinstance(last_layer, nn.Linear):
      nn.init.xavier_normal_(last_layer.weight, gain=0.01)
      nn.init.zeros_(last_layer.bias)

    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.residual_mlp)

    self.is_recurrent = False
    self.memory_size = 0

  @property
  def cnns(self):
    return None

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    obs_list = [obs[g] for g in self.obs_groups]
    full_obs = torch.cat(obs_list, dim=-1)
    # Frozen base: stochastic sample
    base_obs = full_obs[:, :self.base_obs_dim]
    with torch.no_grad():
      base_action = self.base(base_obs)
    self._last_base_action = base_action

    # Residual input = normalized full obs + base action. Symmetric with the
    # base's Stage 1 ClippedMLPModel: clamp normalized obs to [-clip, clip]
    # before the residual MLP sees them.
    obs_norm = self.obs_normalizer(full_obs)
    if self.obs_clip > 0:
      obs_norm = torch.clamp(obs_norm, -self.obs_clip, self.obs_clip)
    return torch.cat([obs_norm, base_action], dim=-1)

  def forward(
    self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output: bool = False,
  ) -> torch.Tensor:
    latent = self.get_latent(obs, masks, hidden_state)
    residual = self.residual_mlp(latent)  # (B, n_dofs)
    # Classical residual composition. `self._last_base_action` was stored by
    # `get_latent` and is already detached (frozen base under torch.no_grad).
    applied_mean = self._last_base_action + residual * self._residual_action_scale

    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(applied_mean)
        return self.distribution.sample()
      return self.distribution.deterministic_output(applied_mean)
    return applied_mean

  def train(self, mode: bool = True):
    super().train(mode)
    self.base.train(False)
    return self

  def state_dict(self, *args, **kwargs):
    full_sd = super().state_dict(*args, **kwargs)
    return {k: v for k, v in full_sd.items() if not k.startswith("base.")}

  def load_state_dict(self, state_dict, strict=False):
    return super().load_state_dict(state_dict, strict=False)

  @property
  def output_distribution_params(self):
    return self.distribution.params

  @property
  def output_entropy(self):
    return self.distribution.entropy

  @property
  def output_mean(self):
    return self.distribution.mean

  @property
  def output_std(self):
    return self.distribution.std

  def get_hidden_state(self):
    return None

  def detach_hidden_state(self, dones=None):
    pass

  def reset(self, dones=None, hidden_state=None):
    pass

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      obs_list = [obs[g] for g in self.obs_groups]
      mlp_obs = torch.cat(obs_list, dim=-1)
      self.obs_normalizer.update(mlp_obs)

  def get_kl_divergence(self, old_params, new_params):
    return self.distribution.kl_divergence(old_params, new_params)

  def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    return self.distribution.log_prob(outputs)
