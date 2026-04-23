"""ManipTrans command term: loads motion.npz and provides MANO tracking targets.

Unlike mjlab's MotionCommand (which tracks robot vs reference robot), ManipTrans
tracks robot body/site positions against MANO *human* hand joint positions.
Robot retarget data is used for warm-start reset only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  matrix_from_quat,
  quat_conjugate,
  quat_error_magnitude,
  quat_from_matrix,
  quat_mul,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


class ManipTransMotionData:
  """Loads one or more motion.npz files and organizes robot + MANO data.

  Single- and multi-trajectory loading share the same code path. All
  per-frame tensors are stacked into shape (M, T_max, ...) where M is the
  number of trajectories and T_max the longest trajectory length. Shorter
  trajectories are tail-padded by replicating their final frame along the
  time axis; ManipTransCommand double-indexes with (env_traj_idx, time_steps)
  and clamps to per-traj length, so padded tail frames are never read in
  normal rollouts (the padding just guards against pathological reads).

  All trajectories in a multi-traj run must share the same MANO joint
  layout (joint_names identical per side) and the same presence of optional
  fields (obj_*, contact_*, tips_distance are all-or-none across trajs).
  """

  def __init__(
    self,
    motion_file: str | list[str],
    sides: tuple[str, ...],
    n_hand_dofs: int,
    device: str,
  ) -> None:
    motion_files = (
      [motion_file] if isinstance(motion_file, (str, bytes)) else list(motion_file)
    )
    if len(motion_files) == 0:
      raise ValueError("motion_file must be a non-empty path or list of paths")

    datas = [np.load(p, allow_pickle=True) for p in motion_files]
    M = len(datas)
    time_lens = [int(d["joint_pos"].shape[0]) for d in datas]
    T_max = max(time_lens)

    self.num_trajectories: int = M
    self.time_step_totals: torch.Tensor = torch.tensor(
      time_lens, dtype=torch.long, device=device
    )

    # Robot data for warm-start reset.
    # Motion.npz joint order: [right_hand, left_hand, objects].
    # For right-only: take first n_hand_dofs.
    # For left-only: skip right hand joints (offset = n_hand_dofs).
    # For bimanual: take first n_hand_dofs (= right + left).
    joint_offset = n_hand_dofs if sides == ("left",) else 0
    self.joint_pos = self._stack_pad(
      [d["joint_pos"][:, joint_offset:joint_offset + n_hand_dofs] for d in datas],
      T_max, device,
    )
    self.joint_vel = self._stack_pad(
      [d["joint_vel"][:, joint_offset:joint_offset + n_hand_dofs] for d in datas],
      T_max, device,
    )

    # Per-side MANO data for tracking targets
    self.wrist_pos: dict[str, torch.Tensor] = {}
    self.wrist_rot: dict[str, torch.Tensor] = {}
    self.wrist_vel: dict[str, torch.Tensor] = {}
    self.wrist_angvel: dict[str, torch.Tensor] = {}
    self.joints: dict[str, torch.Tensor] = {}
    self.joints_vel: dict[str, torch.Tensor] = {}
    self.joint_names: dict[str, list[str]] = {}
    self.tip_indices: dict[str, torch.Tensor] = {}

    for side in sides:
      prefix = f"mano_{side}_"
      self.wrist_pos[side] = self._stack_pad(
        [d[prefix + "wrist_pos"] for d in datas], T_max, device,
      )
      self.wrist_rot[side] = self._stack_pad(
        [d[prefix + "wrist_rot"] for d in datas], T_max, device,
      )
      self.wrist_vel[side] = self._stack_pad(
        [d[prefix + "wrist_vel"] for d in datas], T_max, device,
      )
      self.wrist_angvel[side] = self._stack_pad(
        [d[prefix + "wrist_angvel"] for d in datas], T_max, device,
      )
      self.joints[side] = self._stack_pad(
        [d[prefix + "joints"] for d in datas], T_max, device,
      )
      self.joints_vel[side] = self._stack_pad(
        [d[prefix + "joints_vel"] for d in datas], T_max, device,
      )

      # Joint names must be identical across trajs so tip_indices and the
      # level1/2/all body-to-MANO indices built by ManipTransCommand stay
      # valid regardless of which trajectory an env is currently running.
      names_lists = [list(d[prefix + "joint_names"]) for d in datas]
      base_names = names_lists[0]
      for i, names in enumerate(names_lists[1:], start=1):
        if names != base_names:
          raise ValueError(
            f"joint_names mismatch for side {side!r}: traj 0 vs traj {i} "
            f"({motion_files[0]} vs {motion_files[i]}). All motion files "
            f"in a multi-trajectory run must share the same MANO joint layout."
          )
      self.joint_names[side] = base_names
      tip_idx = [base_names.index(f"{finger}_tip") for finger in FINGER_NAMES]
      self.tip_indices[side] = torch.tensor(
        tip_idx, dtype=torch.long, device=device
      )

    # Per-side tips_distance: MANO tip to object surface (precomputed)
    self.tips_distance: dict[str, torch.Tensor] = {}
    for side in sides:
      key = f"tips_distance_{side}"
      present = [key in d for d in datas]
      if any(present):
        self._require_all_or_none(present, key, motion_files)
        self.tips_distance[side] = self._stack_pad(
          [d[key] for d in datas], T_max, device,
        )

    # Per-side per-frame contact position on object (object-local frame)
    # Shape: (M, T_max, 5, 3). contact[m, t, i] = target contact point on
    # object for finger i at frame t of trajectory m.
    self.contact_pos_full: dict[str, torch.Tensor] = {}
    self.contact_flags: dict[str, torch.Tensor] = {}
    for side in sides:
      key = f"contact_contact_pos_full_{side}"
      present = [key in d for d in datas]
      if any(present):
        self._require_all_or_none(present, key, motion_files)
        self.contact_pos_full[side] = self._stack_pad(
          [d[key] for d in datas], T_max, device,
        )
      flag_key = f"contact_contact_{side}"
      present_f = [flag_key in d for d in datas]
      if any(present_f):
        self._require_all_or_none(present_f, flag_key, motion_files)
        self.contact_flags[side] = self._stack_pad(
          [d[flag_key] for d in datas], T_max, device,
        )

    # Per-side object trajectory data
    self.obj_pos: dict[str, torch.Tensor] = {}
    self.obj_rotmat: dict[str, torch.Tensor] = {}
    self.obj_vel: dict[str, torch.Tensor] = {}
    self.obj_angvel: dict[str, torch.Tensor] = {}
    # Intrinsic XYZ Euler (α, β, γ) matching MuJoCo 3-hinge composition
    # [rot_x, rot_y, rot_z]. Computed from obj_rotmat and unwrapped across
    # time so the ctrl target is continuous (no ±π canonicalization jumps
    # that would otherwise send the position actuator spinning 360°).
    # Used only by pin_mode="actuated". Shape: (M, T_max, 3).
    self.obj_euler: dict[str, torch.Tensor] = {}
    # Hinge joint velocities = d/dt of the unwrapped Euler. Populated by the
    # command term once env step_dt is known (this class has no env ref).
    # Shape: (M, T_max, 3).
    self.obj_euler_vel: dict[str, torch.Tensor] = {}

    for side in sides:
      prefix = f"obj_{side}_"
      key_pos = prefix + "pos"
      present = [key_pos in d for d in datas]
      if not any(present):
        continue
      self._require_all_or_none(present, key_pos, motion_files)

      self.obj_pos[side] = self._stack_pad(
        [d[key_pos] for d in datas], T_max, device,
      )
      rotmats_np = [d[prefix + "rotmat"] for d in datas]  # each (T_m, 3, 3)
      self.obj_rotmat[side] = self._stack_pad(rotmats_np, T_max, device)
      self.obj_vel[side] = self._stack_pad(
        [d[prefix + "vel"] for d in datas], T_max, device,
      )
      self.obj_angvel[side] = self._stack_pad(
        [d[prefix + "angvel"] for d in datas], T_max, device,
      )

      # Intrinsic XYZ euler from rotmat: R = R_x(α) R_y(β) R_z(γ)
      # → β = asin(R[0,2]), α = atan2(-R[1,2], R[2,2]), γ = atan2(-R[0,1], R[0,0])
      # Unwrap along time per-trajectory on the valid (non-padded) prefix so
      # the replicate pad doesn't pollute ±π jumps across the padding seam.
      euler_list = []
      for R, T_m in zip(rotmats_np, time_lens):
        R_valid = R[:T_m]
        beta = np.arcsin(np.clip(R_valid[:, 0, 2], -0.9999, 0.9999))
        alpha = np.arctan2(-R_valid[:, 1, 2], R_valid[:, 2, 2])
        gamma = np.arctan2(-R_valid[:, 0, 1], R_valid[:, 0, 0])
        euler = np.stack([alpha, beta, gamma], axis=-1)  # (T_m, 3)
        euler = np.unwrap(euler, axis=0)
        euler_list.append(euler)
      self.obj_euler[side] = self._stack_pad(euler_list, T_max, device)

  @staticmethod
  def _stack_pad(
    arrays: list[np.ndarray],
    T_max: int,
    device: str,
    dtype: torch.dtype = torch.float32,
  ) -> torch.Tensor:
    """Stack per-traj arrays `(T_m, *rest)` into `(M, T_max, *rest)` with
    replicate padding along the time axis (tail frames copied from the last
    valid frame). Padding is defensive; callers clamp reads by per-traj
    length so pad frames are not seen during normal rollouts.
    """
    padded = []
    for arr in arrays:
      T_m = arr.shape[0]
      if T_m == T_max:
        padded.append(arr)
        continue
      last = arr[-1:]
      pad = np.broadcast_to(last, (T_max - T_m, *arr.shape[1:]))
      padded.append(np.concatenate([arr, pad], axis=0))
    stacked = np.stack(padded, axis=0)
    return torch.tensor(stacked, dtype=dtype, device=device)

  @staticmethod
  def _require_all_or_none(
    present: list[bool], key: str, motion_files: list[str]
  ) -> None:
    if not all(present):
      havers = [motion_files[i] for i, p in enumerate(present) if p]
      missing = [motion_files[i] for i, p in enumerate(present) if not p]
      raise ValueError(
        f"Field {key!r} is present in some motion files but not others — "
        f"all-or-none required across a multi-trajectory run.\n"
        f"  present in: {havers}\n  missing in: {missing}"
      )


class ManipTransCommand(CommandTerm):
  """Command term for ManipTrans dexterous manipulation.

  Loads motion.npz with robot (warm-start) and MANO (tracking targets).
  Robot joint data initializes the hand on reset.
  MANO wrist/joint data provides tracking targets for rewards.
  """

  cfg: ManipTransCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: ManipTransCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    n_hand_dofs = self.robot.num_joints

    self.motion = ManipTransMotionData(
      cfg.motion_file, cfg.sides, n_hand_dofs, device=self.device
    )

    # Populate per-side hinge-joint velocity trajectories now that env step
    # dt is known. Used at reset to give actuated objects a non-cold-start
    # initial velocity. Stored on `self.motion` for symmetry with obj_euler.
    # Computed per-trajectory on each traj's valid (non-padded) prefix so the
    # gradient at the pad seam doesn't leak into the last valid velocity.
    # Shape: (M, T_max, 3); padded tail replicated to match obj_euler.
    step_dt = env.step_dt
    time_lens = self.motion.time_step_totals.tolist()
    T_max = int(self.motion.time_step_totals.max().item())
    for side in cfg.sides:
      if side not in self.motion.obj_euler:
        continue
      euler_all = self.motion.obj_euler[side].cpu().numpy()  # (M, T_max, 3)
      vel_all = np.zeros_like(euler_all)
      for m, T_m in enumerate(time_lens):
        vel_valid = np.gradient(euler_all[m, :T_m], axis=0) / step_dt  # (T_m, 3)
        vel_all[m, :T_m] = vel_valid
        if T_m < T_max:
          vel_all[m, T_m:] = vel_valid[-1:]
      self.motion.obj_euler_vel[side] = torch.tensor(
        vel_all, dtype=torch.float32, device=self.device
      )

    # Per-side: wrist body index and tip site indices
    self._side_list = list(cfg.sides)
    self._wrist_body_indices: dict[str, int] = {}
    self._tip_site_indices: dict[str, list[int]] = {}

    self._contact_site_indices: dict[str, list[int]] = {}
    for side in cfg.sides:
      self._wrist_body_indices[side] = self.robot.body_names.index(
        cfg.wrist_body_names[side]
      )
      tip_names = [f"track_hand_{side}_{finger}_tip" for finger in FINGER_NAMES]
      ids, _ = self.robot.find_sites(tip_names, preserve_order=True)
      self._tip_site_indices[side] = ids
      contact_names = [f"contact_{side}_{finger}_tip" for finger in FINGER_NAMES]
      contact_ids, _ = self.robot.find_sites(contact_names, preserve_order=True)
      self._contact_site_indices[side] = contact_ids

    # All tracked bodies (per side, matching the hand's body_names minus wrist).
    # Each maps to a MANO joint for delta obs and tracking rewards.
    # Per-hand mapping is supplied via `cfg.body_mapping` from the hand's
    # asset_zoo constants module (e.g. asset_zoo.hands.xhand.constants.BODY_MAPPING).
    _ALL_BODIES_MANO = cfg.body_mapping["all"]

    # Level 1/2 for rewards (one body per finger per level).
    _LEVEL1_BODIES = cfg.body_mapping["level1"]
    _LEVEL2_BODIES = cfg.body_mapping["level2"]
    _LEVEL1_MANO = {
      "thumb": "thumb_proximal",
      "index": "index_proximal",
      "middle": "middle_proximal",
      "ring": "ring_proximal",
      "pinky": "pinky_proximal",
    }
    _LEVEL2_MANO = {
      "thumb": "thumb_intermediate",
      "index": "index_intermediate",
      "middle": "middle_intermediate",
      "ring": "ring_intermediate",
      "pinky": "pinky_intermediate",
    }

    self._level1_body_indices: dict[str, list[int]] = {}
    self._level1_mano_indices: dict[str, list[int]] = {}
    self._level2_body_indices: dict[str, list[int]] = {}
    self._level2_mano_indices: dict[str, list[int]] = {}

    for side in cfg.sides:
      l1_body = []
      l1_mano = []
      l2_body = []
      l2_mano = []
      joint_names = self.motion.joint_names[side]
      for finger in FINGER_NAMES:
        # Level 1
        body_name = f"{side}_{_LEVEL1_BODIES[finger]}"
        body_idx = self.robot.body_names.index(body_name)
        mano_idx = joint_names.index(_LEVEL1_MANO[finger])
        l1_body.append(body_idx)
        l1_mano.append(mano_idx)
        # Level 2
        body_name = f"{side}_{_LEVEL2_BODIES[finger]}"
        body_idx = self.robot.body_names.index(body_name)
        mano_idx = joint_names.index(_LEVEL2_MANO[finger])
        l2_body.append(body_idx)
        l2_mano.append(mano_idx)

      self._level1_body_indices[side] = l1_body
      self._level1_mano_indices[side] = l1_mano
      self._level2_body_indices[side] = l2_body
      self._level2_mano_indices[side] = l2_mano

    # All 12 non-tip body-to-MANO pairs per side (for full joint delta obs)
    self._all_body_indices: dict[str, list[int]] = {}
    self._all_mano_indices: dict[str, list[int]] = {}
    for side in cfg.sides:
      body_ids = []
      mano_ids = []
      joint_names = self.motion.joint_names[side]
      for robot_body, mano_joint in _ALL_BODIES_MANO:
        body_name = f"{side}_{robot_body}"
        body_ids.append(self.robot.body_names.index(body_name))
        mano_ids.append(joint_names.index(mano_joint))
      self._all_body_indices[side] = body_ids
      self._all_mano_indices[side] = mano_ids

    # Resolve wrist (translation + rotation) and finger joint IDs for noise
    wrist_ids, wrist_names = self.robot.find_joints_by_actuator_names(
      (".*forearm.*|.*pos_[xyz].*|.*rot_[xyz].*",)
    )
    finger_ids, _ = self.robot.find_joints_by_actuator_names(
      (".*thumb.*", ".*index.*", ".*mid.*", ".*ring.*", ".*pinky.*")
    )
    # Split wrist into translation (tx/ty/tz) and rotation (roll/pitch/yaw)
    wrist_trans_ids = [i for i, n in zip(wrist_ids, wrist_names) if "tx" in n or "ty" in n or "tz" in n]
    wrist_rot_ids = [i for i, n in zip(wrist_ids, wrist_names) if "roll" in n or "pitch" in n or "yaw" in n]
    self._wrist_trans_ids = torch.tensor(wrist_trans_ids, dtype=torch.long, device=self.device)
    self._wrist_rot_ids = torch.tensor(wrist_rot_ids, dtype=torch.long, device=self.device)
    self._wrist_joint_ids = torch.tensor(wrist_ids, dtype=torch.long, device=self.device)
    self._finger_joint_ids = torch.tensor(finger_ids, dtype=torch.long, device=self.device)

    # Time stepping + per-env trajectory assignment. `env_traj_idx[i]` is
    # the trajectory (index into motion.num_trajectories) that env i is
    # currently running; it is uniformly resampled on every reset. Single-
    # traj runs keep it at 0 forever (M=1, trivially correct).
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.env_traj_idx = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    # Adaptive sampling state — per-trajectory bins. `bin_count` is sized to
    # the longest trajectory (bins beyond each traj's own length are never
    # read because sampling clamps by `bins_per_traj`). Time constant: each
    # bin represents ~1 second of motion (`1 / env.step_dt` frames per bin).
    M = self.motion.num_trajectories
    self.bin_count = int(
      int(self.motion.time_step_totals.max().item()) // (1 / env.step_dt)
    ) + 1
    # Valid bin count per trajectory (sampling restricts to [0, bins_per_traj[m])).
    self.bins_per_traj = (
      self.motion.time_step_totals.float() // (1 / env.step_dt)
    ).long() + 1
    self.bin_failed_count = torch.zeros(
      M, self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      M, self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
      [cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    # Metrics
    self.metrics["error_wrist_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_wrist_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_tip_pos"] = torch.zeros(self.num_envs, device=self.device)
    if cfg.object_entity_names is not None:
      self.metrics["error_obj_pos"] = torch.zeros(self.num_envs, device=self.device)

    # Contact metrics — episode-level "peak" aggregations use stateful
    # running-max buffers reset in _resample_command; per-step "integrated"
    # and "frac" metrics rely on framework episode-mean to compose. Sensors
    # are accessed by name in _update_metrics; if they're absent (e.g.
    # --no_object path) the writer skips without error.
    self._ep_force_peak = torch.zeros(
      self.num_envs, len(self._side_list), 5, device=self.device
    )
    self._ep_depth_peak = torch.zeros(
      self.num_envs, len(self._side_list), 5, device=self.device
    )
    side_prefixes = {"right": "r", "left": "l"}
    for side in cfg.sides:
      p = side_prefixes[side]
      for finger in FINGER_NAMES:
        for key in ("force_peak", "depth_peak",
                    "contact_frac", "force_integrated", "depth_integrated",
                    "ref_flag_frac"):
          self.metrics[f"{key}_{p}_{finger}"] = torch.zeros(
            self.num_envs, device=self.device
          )

    # Pinning metrics (adaptive-pin era): per-side stateful peak buffers for
    # object pos/rot deviation, + per-step pin_fired buffer consumed by the
    # pin_penalty reward and the pin_fire_frac / pin_fire_count metrics.
    n_sides = len(self._side_list)
    self._ep_pos_dev_max = torch.zeros(
      self.num_envs, n_sides, device=self.device
    )
    self._ep_rot_dev_max = torch.zeros(
      self.num_envs, n_sides, device=self.device
    )
    self._ep_pin_fire_count = torch.zeros(
      self.num_envs, n_sides, device=self.device
    )
    # Exposed per-step for the pin_penalty reward function.
    self.pin_fired_this_step = torch.zeros(
      self.num_envs, n_sides, dtype=torch.bool, device=self.device
    )
    for side in cfg.sides:
      p = side_prefixes[side]
      for key in ("pos_dev_max", "rot_dev_max", "pin_fire_count", "pin_fire_frac"):
        self.metrics[f"{key}_{p}"] = torch.zeros(self.num_envs, device=self.device)

  # --- Command property ---

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.ref_joint_pos, self.ref_joint_vel], dim=1)

  # --- Reference joint state (for reset) ---

  @property
  def ref_joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.env_traj_idx, self.time_steps]

  @property
  def ref_joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.env_traj_idx, self.time_steps]

  # --- MANO tracking targets ---

  @property
  def mano_wrist_pos_w(self) -> torch.Tensor:
    """MANO wrist positions. Shape: (B, n_sides, 3)."""
    parts = []
    for side in self._side_list:
      pos = self.motion.wrist_pos[side][self.env_traj_idx, self.time_steps]  # (B, 3)
      pos = pos + self._env.scene.env_origins
      parts.append(pos)
    return torch.stack(parts, dim=1)

  @property
  def mano_wrist_rot_w(self) -> torch.Tensor:
    """MANO wrist rotation matrices. Shape: (B, n_sides, 3, 3)."""
    parts = []
    for side in self._side_list:
      parts.append(self.motion.wrist_rot[side][self.env_traj_idx, self.time_steps])
    return torch.stack(parts, dim=1)

  @property
  def mano_wrist_quat_w(self) -> torch.Tensor:
    """MANO wrist quaternions (from rotmat). Shape: (B, n_sides, 4)."""
    return quat_from_matrix(self.mano_wrist_rot_w)

  @property
  def mano_wrist_vel_w(self) -> torch.Tensor:
    """MANO wrist linear velocities. Shape: (B, n_sides, 3)."""
    parts = []
    for side in self._side_list:
      parts.append(self.motion.wrist_vel[side][self.env_traj_idx, self.time_steps])
    return torch.stack(parts, dim=1)

  @property
  def mano_wrist_angvel_w(self) -> torch.Tensor:
    """MANO wrist angular velocities. Shape: (B, n_sides, 3)."""
    parts = []
    for side in self._side_list:
      parts.append(self.motion.wrist_angvel[side][self.env_traj_idx, self.time_steps])
    return torch.stack(parts, dim=1)

  @property
  def mano_tip_pos_w(self) -> torch.Tensor:
    """MANO fingertip positions (5 per side). Shape: (B, n_sides, 5, 3)."""
    parts = []
    for side in self._side_list:
      all_joints = self.motion.joints[side][self.env_traj_idx, self.time_steps]  # (B, 20, 3)
      tips = all_joints[:, self.motion.tip_indices[side]]  # (B, 5, 3)
      tips = tips + self._env.scene.env_origins[:, None, :]
      parts.append(tips)
    return torch.stack(parts, dim=1)

  @property
  def mano_tip_vel_w(self) -> torch.Tensor:
    """MANO fingertip velocities (5 per side). Shape: (B, n_sides, 5, 3)."""
    parts = []
    for side in self._side_list:
      all_vel = self.motion.joints_vel[side][self.env_traj_idx, self.time_steps]  # (B, 20, 3)
      tips = all_vel[:, self.motion.tip_indices[side]]  # (B, 5, 3)
      parts.append(tips)
    return torch.stack(parts, dim=1)

  # --- Robot state ---

  @property
  def robot_wrist_pos_w(self) -> torch.Tensor:
    """Robot wrist body positions. Shape: (B, n_sides, 3)."""
    parts = []
    for side in self._side_list:
      idx = self._wrist_body_indices[side]
      parts.append(self.robot.data.body_link_pos_w[:, idx])
    return torch.stack(parts, dim=1)

  @property
  def robot_wrist_quat_w(self) -> torch.Tensor:
    """Robot wrist body quaternions. Shape: (B, n_sides, 4)."""
    parts = []
    for side in self._side_list:
      idx = self._wrist_body_indices[side]
      parts.append(self.robot.data.body_link_quat_w[:, idx])
    return torch.stack(parts, dim=1)

  @property
  def robot_wrist_rot_w(self) -> torch.Tensor:
    """Robot wrist rotation matrices (from quat). Shape: (B, n_sides, 3, 3)."""
    return matrix_from_quat(self.robot_wrist_quat_w)

  @property
  def robot_wrist_vel_w(self) -> torch.Tensor:
    """Robot wrist linear velocities. Shape: (B, n_sides, 3)."""
    parts = []
    for side in self._side_list:
      idx = self._wrist_body_indices[side]
      parts.append(self.robot.data.body_link_lin_vel_w[:, idx])
    return torch.stack(parts, dim=1)

  @property
  def robot_wrist_angvel_w(self) -> torch.Tensor:
    """Robot wrist angular velocities. Shape: (B, n_sides, 3)."""
    parts = []
    for side in self._side_list:
      idx = self._wrist_body_indices[side]
      parts.append(self.robot.data.body_link_ang_vel_w[:, idx])
    return torch.stack(parts, dim=1)

  @property
  def robot_tip_pos_w(self) -> torch.Tensor:
    """Robot fingertip site positions (5 per side). Shape: (B, n_sides, 5, 3)."""
    parts = []
    for side in self._side_list:
      ids = self._tip_site_indices[side]
      parts.append(self.robot.data.site_pos_w[:, ids])
    return torch.stack(parts, dim=1)

  @property
  def robot_contact_pos_w(self) -> torch.Tensor:
    """Robot contact sensor site positions (5 per side). Shape: (B, n_sides, 5, 3)."""
    parts = []
    for side in self._side_list:
      ids = self._contact_site_indices[side]
      parts.append(self.robot.data.site_pos_w[:, ids])
    return torch.stack(parts, dim=1)

  @property
  def ref_contact_pos_w(self) -> torch.Tensor:
    """Reference contact points on object, in world frame. Shape: (B, n_sides, 5, 3).

    Per-frame target contact positions from preprocessing (object-local frame),
    transformed to world via current sim object pose. Only valid when contact flag is set.
    """
    parts = []
    sim_obj_quat = self.sim_obj_quat_w  # (B, n_sides, 4)
    sim_obj_pos = self.sim_obj_pos_w  # (B, n_sides, 3)
    for side in self._side_list:
      si = self._side_list.index(side)
      local_pts = self.motion.contact_pos_full[side][
        self.env_traj_idx, self.time_steps
      ]  # (B, 5, 3)
      obj_pos = sim_obj_pos[:, si]  # (B, 3)
      obj_rot = matrix_from_quat(sim_obj_quat[:, si])  # (B, 3, 3)
      world_pts = obj_pos[:, None, :] + torch.einsum("bij,bkj->bki", obj_rot, local_pts)
      parts.append(world_pts)
    return torch.stack(parts, dim=1)

  @property
  def ref_contact_flags(self) -> torch.Tensor:
    """Binary contact expected per finger per side. Shape: (B, n_sides, 5)."""
    parts = []
    for side in self._side_list:
      parts.append(
        self.motion.contact_flags[side][self.env_traj_idx, self.time_steps]
      )
    return torch.stack(parts, dim=1)

  @property
  def mano_tips_distance(self) -> torch.Tensor:
    """Precomputed MANO tip-to-object-surface distance. Shape: (B, n_sides, 5)."""
    parts = []
    for side in self._side_list:
      parts.append(
        self.motion.tips_distance[side][self.env_traj_idx, self.time_steps]
      )
    return torch.stack(parts, dim=1)

  # --- All body joint tracking (17 per side: 12 non-tip + 5 tip) ---

  def mano_all_joints_pos_w(self, side: str) -> torch.Tensor:
    """MANO positions for all 12 non-tip joints. Shape: (B, 12, 3)."""
    mano_indices = self._all_mano_indices[side]
    all_joints = self.motion.joints[side][self.env_traj_idx, self.time_steps]  # (B, 20, 3)
    pts = all_joints[:, mano_indices]  # (B, 12, 3)
    return pts + self._env.scene.env_origins[:, None, :]

  def robot_all_joints_pos_w(self, side: str) -> torch.Tensor:
    """Robot positions for all 12 non-tip bodies. Shape: (B, 12, 3)."""
    body_indices = self._all_body_indices[side]
    return self.robot.data.body_link_pos_w[:, body_indices]  # (B, 12, 3)

  def mano_all_joints_vel_w(self, side: str) -> torch.Tensor:
    """MANO velocities for all 12 non-tip joints. Shape: (B, 12, 3)."""
    mano_indices = self._all_mano_indices[side]
    all_vel = self.motion.joints_vel[side][self.env_traj_idx, self.time_steps]  # (B, 20, 3)
    return all_vel[:, mano_indices]  # (B, 12, 3)

  def robot_all_joints_vel_w(self, side: str) -> torch.Tensor:
    """Robot velocities for all 12 non-tip bodies. Shape: (B, 12, 3)."""
    body_indices = self._all_body_indices[side]
    return self.robot.data.body_link_lin_vel_w[:, body_indices]  # (B, 12, 3)

  # --- Level 1/2 joint tracking ---

  def mano_level_pos_w(self, side: str, level: int) -> torch.Tensor:
    """MANO joint positions for level 1 or 2. Shape: (B, 5, 3)."""
    mano_indices = self._level1_mano_indices[side] if level == 1 else self._level2_mano_indices[side]
    all_joints = self.motion.joints[side][self.env_traj_idx, self.time_steps]  # (B, 20, 3)
    pts = all_joints[:, mano_indices]  # (B, 5, 3)
    return pts + self._env.scene.env_origins[:, None, :]

  def robot_level_pos_w(self, side: str, level: int) -> torch.Tensor:
    """Robot body positions for level 1 or 2. Shape: (B, 5, 3)."""
    body_indices = self._level1_body_indices[side] if level == 1 else self._level2_body_indices[side]
    return self.robot.data.body_link_pos_w[:, body_indices]  # (B, 5, 3)

  # --- Object trajectory targets ---

  @property
  def has_objects(self) -> bool:
    return self.cfg.object_entity_names is not None

  @property
  def ref_obj_pos_w(self) -> torch.Tensor:
    """Reference object positions. Shape: (B, n_sides, 3)."""
    parts = []
    for side in self._side_list:
      if side in self.motion.obj_pos:
        pos = self.motion.obj_pos[side][self.env_traj_idx, self.time_steps]
        pos = pos + self._env.scene.env_origins
        parts.append(pos)
    return torch.stack(parts, dim=1)

  @property
  def ref_obj_rotmat_w(self) -> torch.Tensor:
    """Reference object rotation matrices. Shape: (B, n_sides, 3, 3)."""
    parts = []
    for side in self._side_list:
      if side in self.motion.obj_rotmat:
        parts.append(
          self.motion.obj_rotmat[side][self.env_traj_idx, self.time_steps]
        )
    return torch.stack(parts, dim=1)

  @property
  def ref_obj_quat_w(self) -> torch.Tensor:
    """Reference object quaternions. Shape: (B, n_sides, 4)."""
    return quat_from_matrix(self.ref_obj_rotmat_w)

  @property
  def ref_obj_vel_w(self) -> torch.Tensor:
    """Reference object velocities. Shape: (B, n_sides, 3)."""
    parts = []
    for side in self._side_list:
      if side in self.motion.obj_vel:
        parts.append(
          self.motion.obj_vel[side][self.env_traj_idx, self.time_steps]
        )
    return torch.stack(parts, dim=1)

  # --- Sim object state ---

  def _obj_mass_body_idx(self, obj) -> int:
    """Index (within the entity's body_ids list) of the body that carries the
    mass/geoms. For freejoint objects this is the root body (index 0). For
    fixed-base articulated objects (actuated pin_mode), mjlab wraps the entity
    in a mocap_base parent, so the mass body is the last one in body_ids.
    """
    return len(obj.indexing.body_ids) - 1

  @property
  def sim_obj_pos_w(self) -> torch.Tensor:
    """Sim object positions (of the body with mass/geoms). Shape: (B, n_sides, 3)."""
    parts = []
    for side in self._side_list:
      obj_name = self.cfg.object_entity_names[side]
      obj: Entity = self._env.scene[obj_name]
      idx = self._obj_mass_body_idx(obj)
      parts.append(obj.data.body_link_pos_w[:, idx])
    return torch.stack(parts, dim=1)

  @property
  def sim_obj_quat_w(self) -> torch.Tensor:
    """Sim object quaternions (of the body with mass/geoms). Shape: (B, n_sides, 4)."""
    parts = []
    for side in self._side_list:
      obj_name = self.cfg.object_entity_names[side]
      obj: Entity = self._env.scene[obj_name]
      idx = self._obj_mass_body_idx(obj)
      parts.append(obj.data.body_link_quat_w[:, idx])
    return torch.stack(parts, dim=1)

  @property
  def sim_obj_vel_w(self) -> torch.Tensor:
    """Sim object velocities (of the body with mass). Shape: (B, n_sides, 3)."""
    parts = []
    for side in self._side_list:
      obj_name = self.cfg.object_entity_names[side]
      obj: Entity = self._env.scene[obj_name]
      idx = self._obj_mass_body_idx(obj)
      parts.append(obj.data.body_link_lin_vel_w[:, idx])
    return torch.stack(parts, dim=1)

  @property
  def ref_obj_angvel_w(self) -> torch.Tensor:
    """Reference object angular velocities. Shape: (B, n_sides, 3)."""
    parts = []
    for side in self._side_list:
      if side in self.motion.obj_angvel:
        parts.append(
          self.motion.obj_angvel[side][self.env_traj_idx, self.time_steps]
        )
    return torch.stack(parts, dim=1)

  @property
  def sim_obj_angvel_w(self) -> torch.Tensor:
    """Sim object angular velocities (of the body with mass). Shape: (B, n_sides, 3)."""
    parts = []
    for side in self._side_list:
      obj_name = self.cfg.object_entity_names[side]
      obj: Entity = self._env.scene[obj_name]
      idx = self._obj_mass_body_idx(obj)
      parts.append(obj.data.body_link_ang_vel_w[:, idx])
    return torch.stack(parts, dim=1)

  # --- Future trajectory targets (1-step lookahead) ---

  def _next_time_steps(self) -> torch.Tensor:
    """Next frame index per env, clamped to each env's trajectory length."""
    cap = self.motion.time_step_totals[self.env_traj_idx] - 1
    return torch.minimum(self.time_steps + 1, cap)

  @property
  def next_obj_pos_w(self) -> torch.Tensor:
    """Next-frame object positions. Shape: (B, n_sides, 3)."""
    next_t = self._next_time_steps()
    parts = []
    for side in self._side_list:
      if side in self.motion.obj_pos:
        pos = self.motion.obj_pos[side][self.env_traj_idx, next_t] + self._env.scene.env_origins
        parts.append(pos)
    return torch.stack(parts, dim=1)

  @property
  def next_obj_quat_w(self) -> torch.Tensor:
    """Next-frame object quaternions. Shape: (B, n_sides, 4)."""
    next_t = self._next_time_steps()
    parts = []
    for side in self._side_list:
      if side in self.motion.obj_rotmat:
        rotmat = self.motion.obj_rotmat[side][self.env_traj_idx, next_t]
        parts.append(quat_from_matrix(rotmat))
    return torch.stack(parts, dim=1)

  @property
  def next_obj_vel_w(self) -> torch.Tensor:
    """Next-frame object velocities. Shape: (B, n_sides, 3)."""
    next_t = self._next_time_steps()
    parts = []
    for side in self._side_list:
      if side in self.motion.obj_vel:
        parts.append(self.motion.obj_vel[side][self.env_traj_idx, next_t])
    return torch.stack(parts, dim=1)

  # --- CommandTerm abstract methods ---

  def _update_metrics(self) -> None:
    wrist_err = torch.norm(
      self.mano_wrist_pos_w - self.robot_wrist_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_wrist_pos"] = wrist_err

    rot_err = quat_error_magnitude(
      self.mano_wrist_quat_w, self.robot_wrist_quat_w
    )
    self.metrics["error_wrist_rot"] = rot_err.mean(dim=-1)

    tip_err = torch.norm(
      self.mano_tip_pos_w - self.robot_tip_pos_w, dim=-1
    ).mean(dim=(-1, -2))
    self.metrics["error_tip_pos"] = tip_err

    if self.has_objects:
      obj_err = torch.norm(
        self.ref_obj_pos_w - self.sim_obj_pos_w, dim=-1
      ).mean(dim=-1)
      self.metrics["error_obj_pos"] = obj_err

    # Per-finger contact metrics. Reads netforce + mindist sensors by name.
    side_prefixes = {"right": "r", "left": "l"}
    for si, side in enumerate(self._side_list):
      p = side_prefixes[side]
      try:
        force_sensor = self._env.scene[f"{p}_fingertip_contact"]
      except KeyError:
        force_sensor = None
      try:
        pen_sensor = self._env.scene[f"{p}_fingertip_penetration"]
      except KeyError:
        pen_sensor = None

      if force_sensor is not None:
        force_mag = torch.norm(force_sensor.data.force, dim=-1)  # (B, 5)
        self._ep_force_peak[:, si] = torch.maximum(
          self._ep_force_peak[:, si], force_mag
        )

      if pen_sensor is not None:
        depth = torch.clamp(-pen_sensor.data.dist, min=0.0)  # (B, 5)
        found = (pen_sensor.data.found > 0).to(depth.dtype)  # (B, 5)
        self._ep_depth_peak[:, si] = torch.maximum(
          self._ep_depth_peak[:, si], depth
        )

      ref_flag = self.ref_contact_flags[:, si].to(torch.float32)  # (B, 5)

      for fi, finger in enumerate(FINGER_NAMES):
        key_suffix = f"{p}_{finger}"
        self.metrics[f"force_peak_{key_suffix}"] = self._ep_force_peak[:, si, fi]
        self.metrics[f"depth_peak_{key_suffix}"] = self._ep_depth_peak[:, si, fi]
        if force_sensor is not None and pen_sensor is not None:
          self.metrics[f"force_integrated_{key_suffix}"] = (
            force_mag[:, fi] * found[:, fi]
          )
          self.metrics[f"depth_integrated_{key_suffix}"] = (
            depth[:, fi] * found[:, fi]
          )
          self.metrics[f"contact_frac_{key_suffix}"] = found[:, fi]
        self.metrics[f"ref_flag_frac_{key_suffix}"] = ref_flag[:, fi]

    # Pinning metrics (per side). Compute deviations over all envs even when
    # adaptive_pin is off so the metrics are a passive diagnostic under the
    # fixed-interval schedule too.
    if self.has_objects:
      pos_dev = torch.norm(
        self.ref_obj_pos_w - self.sim_obj_pos_w, dim=-1
      )  # (B, n_sides)
      rot_dev = quat_error_magnitude(
        self.ref_obj_quat_w, self.sim_obj_quat_w
      )  # (B, n_sides)
      self._ep_pos_dev_max = torch.maximum(self._ep_pos_dev_max, pos_dev)
      self._ep_rot_dev_max = torch.maximum(self._ep_rot_dev_max, rot_dev)
      pin_fired_f = self.pin_fired_this_step.to(torch.float32)  # (B, n_sides)
      self._ep_pin_fire_count = self._ep_pin_fire_count + pin_fired_f
      for si, side in enumerate(self._side_list):
        p = side_prefixes[side]
        self.metrics[f"pos_dev_max_{p}"] = self._ep_pos_dev_max[:, si]
        self.metrics[f"rot_dev_max_{p}"] = self._ep_rot_dev_max[:, si]
        self.metrics[f"pin_fire_count_{p}"] = self._ep_pin_fire_count[:, si]
        self.metrics[f"pin_fire_frac_{p}"] = pin_fired_f[:, si]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Reset episode-level contact peak buffers on new episode.
    self._ep_force_peak[env_ids] = 0.0
    self._ep_depth_peak[env_ids] = 0.0
    # Reset pinning stateful metric buffers on new episode.
    self._ep_pos_dev_max[env_ids] = 0.0
    self._ep_rot_dev_max[env_ids] = 0.0
    self._ep_pin_fire_count[env_ids] = 0.0

    # Random per-reset trajectory assignment: each resetting env draws a
    # fresh traj_idx uniformly from [0, M). For M=1 this is a no-op.
    if self.motion.num_trajectories > 1:
      self.env_traj_idx[env_ids] = torch.randint(
        0, self.motion.num_trajectories,
        (len(env_ids),), device=self.device, dtype=torch.long,
      )

    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      self._adaptive_sampling(env_ids)

    # Warm-start with ManipTrans-matched noise:
    # - Wrist translation (tx/ty/tz): trajectory + randn * 0.01
    # - Wrist rotation (roll/pitch/yaw): trajectory + randn * pi/18 (10°)
    # - Finger DOFs: trajectory + randn * range / 8
    # - Wrist velocity: trajectory + randn * 0.01
    # - Finger velocity: randn * 0.1
    joint_pos = self.ref_joint_pos[env_ids].clone()
    joint_vel = self.ref_joint_vel[env_ids].clone()

    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]

    # Wrist translation noise: randn * 0.01 (1cm)
    joint_pos[:, self._wrist_trans_ids] += (
      torch.randn(len(env_ids), len(self._wrist_trans_ids), device=self.device) * 0.01
    )
    # Wrist rotation noise: randn * pi/18 (10°)
    joint_pos[:, self._wrist_rot_ids] += (
      torch.randn(len(env_ids), len(self._wrist_rot_ids), device=self.device) * (3.14159265 / 18.0)
    )
    # Finger noise: randn * range / 8
    finger_range = (
      soft_limits[:, self._finger_joint_ids, 1] - soft_limits[:, self._finger_joint_ids, 0]
    )
    joint_pos[:, self._finger_joint_ids] += (
      torch.randn(len(env_ids), len(self._finger_joint_ids), device=self.device)
      * (finger_range / 8.0)
    )
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])

    # Wrist velocity: trajectory + randn * 0.01
    joint_vel[:, self._wrist_joint_ids] += (
      torch.randn(len(env_ids), len(self._wrist_joint_ids), device=self.device) * 0.01
    )
    # Finger velocity: randn * 0.1
    joint_vel[:, self._finger_joint_ids] = (
      torch.randn(len(env_ids), len(self._finger_joint_ids), device=self.device) * 0.1
    )

    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

    # Reset objects to demo trajectory pose
    if self.has_objects:
      pin_mode = getattr(self.cfg, "pin_mode", "hard")
      tr = self.env_traj_idx[env_ids]
      t = self.time_steps[env_ids]
      for side in self._side_list:
        if side not in self.motion.obj_pos:
          continue
        obj_name = self.cfg.object_entity_names[side]
        obj: Entity = self._env.scene[obj_name]
        si = self._side_list.index(side)
        obj_pos = self.ref_obj_pos_w[env_ids][:, si]  # (N, 3)
        obj_quat = self.ref_obj_quat_w[env_ids][:, si]  # (N, 4)
        if pin_mode == "hard" or pin_mode == "xfrc":
          # xfrc shares the freejoint init path with hard: seed object state
          # from reference; xfrc holds it there each step via external wrench
          # instead of overwriting root_state.
          obj_vel = self.ref_obj_vel_w[env_ids][:, si]
          obj_angvel = self.motion.obj_angvel[side][tr, t]
          root_state = torch.cat([obj_pos, obj_quat, obj_vel, obj_angvel], dim=-1)
          obj.write_root_state_to_sim(root_state, env_ids=env_ids)
          obj.reset(env_ids=env_ids)
        elif pin_mode == "actuated":
          # Local-frame (mocap_base child) joint positions. Use precomputed
          # unwrapped intrinsic-XYZ Euler to avoid the ±π canonicalization
          # jump that would send the actuator target spinning 360° between
          # frames (the "light object rotates" bug).
          local_pos = self.motion.obj_pos[side][tr, t]       # (N, 3)
          euler = self.motion.obj_euler[side][tr, t]         # (N, 3)
          joint_pos = torch.cat([local_pos, euler], dim=-1)  # (N, 6)
          # Initial joint velocities to match the reference trajectory (no
          # cold start from rest):
          #   slide qvel = world linear velocity (slide axes are world-aligned)
          #   hinge qvel = d/dt of unwrapped Euler (precomputed)
          lin_vel = self.motion.obj_vel[side][tr, t]         # (N, 3)
          euler_vel = self.motion.obj_euler_vel[side][tr, t]  # (N, 3)
          joint_vel = torch.cat([lin_vel, euler_vel], dim=-1)  # (N, 6)
          obj.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
          # `obj.reset()` zeros joint_pos_target, so seed it AFTER the reset
          # so the first physics step sees the correct target.
          obj.reset(env_ids=env_ids)
          obj.set_joint_position_target(joint_pos, env_ids=env_ids)
        elif pin_mode == "none":
          obj.reset(env_ids=env_ids)

  def _update_command(self) -> None:
    self.time_steps += 1

    # Wrap around envs that exceeded their trajectory's length.
    wrap_ids = torch.where(
      self.time_steps >= self.motion.time_step_totals[self.env_traj_idx]
    )[0]
    if wrap_ids.numel() > 0:
      self._resample_command(wrap_ids)

    # Reset per-step pin buffer; will be overwritten below when pinning fires.
    self.pin_fired_this_step = torch.zeros_like(self.pin_fired_this_step)

    # Pin objects to reference trajectory every step
    if self.cfg.pin_objects and self.has_objects:
      pin_mode = getattr(self.cfg, "pin_mode", "hard")
      if pin_mode == "hard":
        # Fixed temporal cadence OR adaptive deviation gate. Adaptive: pin
        # fires per-side when pos_dev > pin_pos_threshold OR rot_dev >
        # pin_rot_threshold. Skip step 0 (reset already placed the object).
        ep_len = self._env.episode_length_buf
        n_sides = len(self._side_list)
        if self.cfg.adaptive_pin:
          pos_dev = torch.norm(
            self.ref_obj_pos_w - self.sim_obj_pos_w, dim=-1
          )  # (B, n_sides)
          rot_dev = quat_error_magnitude(
            self.ref_obj_quat_w, self.sim_obj_quat_w
          )  # (B, n_sides)
          deviated = (pos_dev > self.cfg.pin_pos_threshold) | (
            rot_dev > self.cfg.pin_rot_threshold
          )
          pin_mask_ps = (ep_len > 0).unsqueeze(-1) & deviated  # (B, n_sides)
        else:
          T = max(1, int(self.cfg.pin_interval))
          interval_mask = (ep_len > 0) & ((ep_len % T) == 0)  # (B,)
          pin_mask_ps = interval_mask.unsqueeze(-1).expand(-1, n_sides)

        self.pin_fired_this_step = pin_mask_ps

        for side in self._side_list:
          if side not in self.motion.obj_pos:
            continue
          si = self._side_list.index(side)
          pin_ids = torch.where(pin_mask_ps[:, si])[0]
          if pin_ids.numel() == 0:
            continue
          tr_pin = self.env_traj_idx[pin_ids]
          t_pin = self.time_steps[pin_ids]
          obj_name = self.cfg.object_entity_names[side]
          obj: Entity = self._env.scene[obj_name]
          obj_pos = self.ref_obj_pos_w[pin_ids, si]
          obj_quat = self.ref_obj_quat_w[pin_ids, si]
          obj_vel = self.motion.obj_vel[side][tr_pin, t_pin]
          obj_angvel = self.motion.obj_angvel[side][tr_pin, t_pin]
          root_state = torch.cat([obj_pos, obj_quat, obj_vel, obj_angvel], dim=-1)
          obj.write_root_state_to_sim(root_state, env_ids=pin_ids)
      elif pin_mode == "actuated":
        # Position-actuator target via `set_joint_position_target` (mjlab's
        # BuiltinActuatorGroup forwards joint_pos_target → data.ctrl each
        # step). Target is entity-local (mocap_base parent supplies
        # env_origin). Rotation from the precomputed unwrapped intrinsic
        # XYZ Euler (no per-step canonicalization jumps).
        for side in self._side_list:
          if side not in self.motion.obj_pos:
            continue
          obj_name = self.cfg.object_entity_names[side]
          obj: Entity = self._env.scene[obj_name]
          local_pos = self.motion.obj_pos[side][self.env_traj_idx, self.time_steps]  # (B, 3)
          euler = self.motion.obj_euler[side][self.env_traj_idx, self.time_steps]  # (B, 3)
          target = torch.cat([local_pos, euler], dim=-1)  # (B, 6)
          obj.set_joint_position_target(target)
      elif pin_mode == "xfrc":
        # DexMachina-style soft PD on the object freejoint via xfrc_applied.
        # The object is a plain freejoint entity (single body, gravity on);
        # we inject a world-frame wrench computed from reference vs. sim error.
        #   force  = kp_pos * (ref_pos  - sim_pos)  + kv_pos * (ref_vel    - sim_vel)
        #   torque = kp_rot * axis_angle(ref_q sim_q^-1) + kv_rot * (ref_w - sim_w)
        # Axis-angle is in world frame (consistent with sim_obj_angvel_w).
        # Gravity is unmodified — PD must carry the weight at steady state.
        kp_pos = self.cfg.xfrc_kp_pos
        kv_pos = self.cfg.xfrc_kv_pos
        kp_rot = self.cfg.xfrc_kp_rot
        kv_rot = self.cfg.xfrc_kv_rot
        sim_pos_all = self.sim_obj_pos_w          # (B, n_sides, 3)
        sim_quat_all = self.sim_obj_quat_w        # (B, n_sides, 4)
        sim_vel_all = self.sim_obj_vel_w          # (B, n_sides, 3)
        sim_angvel_all = self.sim_obj_angvel_w    # (B, n_sides, 3)
        ref_pos_all = self.ref_obj_pos_w          # (B, n_sides, 3)
        ref_quat_all = self.ref_obj_quat_w        # (B, n_sides, 4)
        for side in self._side_list:
          if side not in self.motion.obj_pos:
            continue
          si = self._side_list.index(side)
          obj_name = self.cfg.object_entity_names[side]
          obj: Entity = self._env.scene[obj_name]

          ref_pos = ref_pos_all[:, si]
          ref_quat = ref_quat_all[:, si]
          ref_vel = self.motion.obj_vel[side][self.env_traj_idx, self.time_steps]
          ref_angvel = self.motion.obj_angvel[side][self.env_traj_idx, self.time_steps]
          sim_pos = sim_pos_all[:, si]
          sim_quat = sim_quat_all[:, si]
          sim_vel = sim_vel_all[:, si]
          sim_angvel = sim_angvel_all[:, si]

          # Linear PD
          force = kp_pos * (ref_pos - sim_pos) + kv_pos * (ref_vel - sim_vel)
          # Rotational PD: axis-angle of delta_q = ref_q * sim_q^{-1}
          delta_q = quat_mul(ref_quat, quat_conjugate(sim_quat))
          axis_angle = axis_angle_from_quat(delta_q)
          torque = kp_rot * axis_angle + kv_rot * (ref_angvel - sim_angvel)

          # Freejoint entity has a single body; apply to it. Shape (B, 1, 3).
          obj.write_external_wrench_to_sim(
            forces=force.unsqueeze(1),
            torques=torque.unsqueeze(1),
          )
      elif pin_mode == "none":
        pass
      else:
        raise ValueError(f"unknown pin_mode={pin_mode}")

    # Update adaptive sampling statistics
    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  # --- Sampling helpers ---

  def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
    """Per-trajectory adaptive resampling.

    `self.bin_failed_count` has shape (M, bin_count) — one EMA-smoothed
    failure histogram per trajectory. `bins_per_traj[m]` is the number of
    valid bins for traj m; bins beyond that are masked out so they never get
    sampled and don't distort the kernel-smoothed distribution.
    """
    # --- 1. Scatter this step's failures into (traj, bin). ---
    episode_failed = self._env.termination_manager.terminated[env_ids]
    self._current_bin_failed.zero_()
    if torch.any(episode_failed):
      fail_envs = env_ids[episode_failed]
      fail_tr = self.env_traj_idx[fail_envs]
      fail_t = self.time_steps[fail_envs]
      fail_T_m = self.motion.time_step_totals[fail_tr].clamp_min(1)
      fail_bpt = self.bins_per_traj[fail_tr]
      fail_bin = torch.clamp(
        (fail_t * fail_bpt) // fail_T_m, 0, self.bin_count - 1
      )
      self._current_bin_failed.index_put_(
        (fail_tr, fail_bin),
        torch.ones_like(fail_bin, dtype=torch.float),
        accumulate=True,
      )

    # --- 2. Build per-traj sampling distribution (M, bin_count). ---
    valid_mask = (
      torch.arange(self.bin_count, device=self.device)[None, :]
      < self.bins_per_traj[:, None]
    ).float()  # (M, bin_count)
    probs = (
      self.bin_failed_count
      + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    )
    # Mask before smoothing so invalid bins contribute zero; the kernel's
    # right-edge replicate pad then replicates an already-zero cell and the
    # smoothing stays inside each traj's valid range.
    probs = probs * valid_mask
    probs = torch.nn.functional.pad(
      probs.unsqueeze(1),
      (0, self.cfg.adaptive_kernel_size - 1),
      mode="replicate",
    )
    probs = torch.nn.functional.conv1d(
      probs, self.kernel.view(1, 1, -1)
    ).squeeze(1)  # (M, bin_count)
    probs = probs * valid_mask
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # --- 3. Sample one bin per resetting env from its traj's row. ---
    tr = self.env_traj_idx[env_ids]
    probs_per_env = probs[tr]  # (N, bin_count)
    sampled_bins = torch.multinomial(
      probs_per_env, 1, replacement=True
    ).squeeze(-1)  # (N,)

    # --- 4. Convert bin → frame using per-env traj length. ---
    bpt = self.bins_per_traj[tr].float()
    T_m = self.motion.time_step_totals[tr].float()
    self.time_steps[env_ids] = (
      (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
      / bpt
      * (T_m - 1)
    ).long()

  def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
    # ManipTrans: floor(T_m * 0.99 * rand) per env, using each env's own
    # trajectory length (self.env_traj_idx was already sampled by
    # _resample_command before this helper runs).
    tr = self.env_traj_idx[env_ids]
    T_m = self.motion.time_step_totals[tr].float()
    self.time_steps[env_ids] = (
      torch.rand(len(env_ids), device=self.device) * 0.99 * T_m
    ).long()


@dataclass(kw_only=True)
class ManipTransCommandCfg(CommandTermCfg):
  """Configuration for ManipTrans motion command."""

  motion_file: "str | list[str]"
  entity_name: str
  sides: tuple[str, ...]
  wrist_body_names: dict[str, str]
  body_mapping: dict
  """Per-hand body-name mapping. Keys: 'all' (sequence of (body, mano_joint)
  pairs for all tracked non-tip bodies), 'level1' and 'level2' (dicts from
  finger name to the body used for the L1/L2 tracking reward). Supplied by
  the hand's asset_zoo constants module (e.g. BODY_MAPPING in
  asset_zoo.hands.xhand.constants). Side prefix is applied at lookup time."""
  object_entity_names: dict[str, str] = None  # side → entity name, e.g. {"right": "object_right"}
  pin_objects: bool = False  # If True, override object root state every step to match reference trajectory
  pin_mode: Literal["hard", "actuated", "xfrc", "none"] = "hard"
  """How to hold the object when `pin_objects=True`.
  - "hard": overwrite object root state to reference every step (original path).
  - "actuated": write object actuator ctrl to reference every step. The object body
    must have 6 position actuators (created via `get_actuated_object_cfg`).
  - "xfrc": DexMachina-style soft PD on the freejoint via `xfrc_applied`. Force
    and torque are computed from world-frame position/rotation error and written
    to the object's mass body each step. Object is a plain freejoint entity
    (created via `get_object_cfg`); gravity is unmodified. Decayable gains via
    `xfrc_kp_pos` / `xfrc_kv_pos` / `xfrc_kp_rot` / `xfrc_kv_rot`.
  - "none": no pinning, even if `pin_objects=True`. Use for Stage 2 curriculum end.
  """
  xfrc_kp_pos: float = 0.0
  xfrc_kv_pos: float = 0.0
  xfrc_kp_rot: float = 0.0
  xfrc_kv_rot: float = 0.0
  pin_interval: int = 6
  """For `pin_mode="hard"`: fixed temporal pin interval T. `ep_len % T == 0`
  gating. T=1 is full hard pin (original behavior). T=N lets the object drift
  N-1 steps between snaps. T >= episode_length is effectively no pin. Ignored
  when `adaptive_pin=True`.
  """
  adaptive_pin: bool = False
  """If True, replace the fixed-interval pin schedule with a deviation-gated
  rule: pin fires only when the sim object has drifted past
  `pin_pos_threshold` OR `pin_rot_threshold` from the reference. Design 1
  ("pure adaptive") — no interval fallback. Only wired for pin_mode="hard".
  """
  pin_pos_threshold: float = 0.030
  """Position deviation threshold (m) for adaptive pinning. Default 3 cm."""
  pin_rot_threshold: float = 1.5708
  """Rotation deviation threshold (rad) for adaptive pinning. Default 90°."""
  joint_position_range: tuple[float, float] = (0.0, 0.0)
  sampling_mode: Literal["adaptive", "uniform", "start"] = "uniform"
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001

  def build(self, env: ManagerBasedRlEnv) -> ManipTransCommand:
    return ManipTransCommand(self, env)
