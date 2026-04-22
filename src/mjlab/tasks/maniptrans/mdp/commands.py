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
  """Loads motion.npz and organizes robot + MANO data.

  Supports multi-motion via ``from_multiple()``: N single-motion datasets are
  concatenated along the frame axis (ProtoMotions-style). Per-motion metadata
  (``num_motions``, ``motion_num_frames``, ``length_starts``) enable O(1)
  per-env dispatch with ``flat_idx = length_starts[motion_id] + frame_idx``.
  When loaded from a single file, ``num_motions == 1`` and the flat index
  degenerates to just the frame index (backward-compatible).
  """

  # --- Multi-motion metadata (set by from_multiple, or defaults for single) ---
  num_motions: int = 1
  motion_num_frames: torch.Tensor | None = None  # (num_motions,)
  length_starts: torch.Tensor | None = None       # (num_motions,) cumulative offsets

  def __init__(
    self,
    motion_file: str,
    sides: tuple[str, ...],
    n_hand_dofs: int,
    device: str,
  ) -> None:
    data = np.load(motion_file, allow_pickle=True)

    joint_offset = n_hand_dofs if sides == ("left",) else 0
    self.joint_pos = torch.tensor(
      data["joint_pos"][:, joint_offset:joint_offset + n_hand_dofs],
      dtype=torch.float32, device=device,
    )
    self.joint_vel = torch.tensor(
      data["joint_vel"][:, joint_offset:joint_offset + n_hand_dofs],
      dtype=torch.float32, device=device,
    )
    self.time_step_total = self.joint_pos.shape[0]
    # Single-motion defaults
    self.num_motions = 1
    self.motion_num_frames = torch.tensor([self.time_step_total], dtype=torch.long, device=device)
    self.length_starts = torch.tensor([0], dtype=torch.long, device=device)

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
      self.wrist_pos[side] = torch.tensor(
        data[prefix + "wrist_pos"], dtype=torch.float32, device=device
      )
      self.wrist_rot[side] = torch.tensor(
        data[prefix + "wrist_rot"], dtype=torch.float32, device=device
      )
      self.wrist_vel[side] = torch.tensor(
        data[prefix + "wrist_vel"], dtype=torch.float32, device=device
      )
      self.wrist_angvel[side] = torch.tensor(
        data[prefix + "wrist_angvel"], dtype=torch.float32, device=device
      )
      self.joints[side] = torch.tensor(
        data[prefix + "joints"], dtype=torch.float32, device=device
      )
      self.joints_vel[side] = torch.tensor(
        data[prefix + "joints_vel"], dtype=torch.float32, device=device
      )

      names = list(data[prefix + "joint_names"])
      self.joint_names[side] = names

      # Find tip indices in the 20-joint array
      tip_idx = []
      for finger in FINGER_NAMES:
        tip_idx.append(names.index(f"{finger}_tip"))
      self.tip_indices[side] = torch.tensor(tip_idx, dtype=torch.long, device=device)

    # Per-side tips_distance: MANO tip to object surface (precomputed)
    self.tips_distance: dict[str, torch.Tensor] = {}
    for side in sides:
      key = f"tips_distance_{side}"
      if key in data:
        self.tips_distance[side] = torch.tensor(
          data[key], dtype=torch.float32, device=device
        )

    # Per-side per-frame contact position on object (object-local frame)
    # Shape: (T, 5, 3). contact[t, i] = target contact point on object for finger i at frame t.
    self.contact_pos_full: dict[str, torch.Tensor] = {}
    self.contact_flags: dict[str, torch.Tensor] = {}
    for side in sides:
      key = f"contact_contact_pos_full_{side}"
      if key in data:
        self.contact_pos_full[side] = torch.tensor(
          data[key], dtype=torch.float32, device=device
        )
      flag_key = f"contact_contact_{side}"
      if flag_key in data:
        self.contact_flags[side] = torch.tensor(
          data[flag_key], dtype=torch.float32, device=device
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
    # Used only by pin_mode="actuated".
    self.obj_euler: dict[str, torch.Tensor] = {}
    # Hinge joint velocities = d/dt of the unwrapped Euler. Populated by the
    # command term once env step_dt is known (not here — this class has no
    # env reference). Shape: (T, 3).
    self.obj_euler_vel: dict[str, torch.Tensor] = {}

    for side in sides:
      prefix = f"obj_{side}_"
      if prefix + "pos" in data:
        self.obj_pos[side] = torch.tensor(
          data[prefix + "pos"], dtype=torch.float32, device=device
        )
        R = data[prefix + "rotmat"]  # (T, 3, 3) numpy
        self.obj_rotmat[side] = torch.tensor(R, dtype=torch.float32, device=device)
        self.obj_vel[side] = torch.tensor(
          data[prefix + "vel"], dtype=torch.float32, device=device
        )
        self.obj_angvel[side] = torch.tensor(
          data[prefix + "angvel"], dtype=torch.float32, device=device
        )
        # Intrinsic XYZ euler from rotmat: R = R_x(α) R_y(β) R_z(γ)
        # → β = asin(R[0,2]), α = atan2(-R[1,2], R[2,2]), γ = atan2(-R[0,1], R[0,0])
        beta = np.arcsin(np.clip(R[:, 0, 2], -0.9999, 0.9999))
        alpha = np.arctan2(-R[:, 1, 2], R[:, 2, 2])
        gamma = np.arctan2(-R[:, 0, 1], R[:, 0, 0])
        euler = np.stack([alpha, beta, gamma], axis=-1)  # (T, 3)
        euler = np.unwrap(euler, axis=0)  # unwrap each axis across time
        self.obj_euler[side] = torch.tensor(
          euler, dtype=torch.float32, device=device
        )

  @classmethod
  def from_multiple(
    cls,
    motion_files: list[str],
    sides: tuple[str, ...],
    n_hand_dofs: int,
    device: str,
  ) -> "ManipTransMotionData":
    """Concatenate N single-motion datasets into one with offset indexing.

    All motions must share the same sides, n_hand_dofs, and MANO joint_names.
    Different-length motions are natively supported via per-motion
    ``motion_num_frames`` / ``length_starts``. Access motion_id=i, frame=t
    via flat index ``length_starts[i] + t``.
    """
    singles = [cls(f, sides, n_hand_dofs, device) for f in motion_files]
    combined = object.__new__(cls)

    # Concat frame-indexed tensors along dim 0
    combined.joint_pos = torch.cat([m.joint_pos for m in singles], dim=0)
    combined.joint_vel = torch.cat([m.joint_vel for m in singles], dim=0)
    combined.time_step_total = combined.joint_pos.shape[0]

    # Per-side dicts: concat each side's tensor
    for attr in ("wrist_pos", "wrist_rot", "wrist_vel", "wrist_angvel",
                 "joints", "joints_vel", "tips_distance", "contact_pos_full",
                 "contact_flags", "obj_pos", "obj_rotmat", "obj_vel",
                 "obj_angvel", "obj_euler"):
      d: dict[str, torch.Tensor] = {}
      for side in sides:
        parts = [getattr(m, attr).get(side) for m in singles if side in getattr(m, attr)]
        if parts and all(p is not None for p in parts):
          d[side] = torch.cat(parts, dim=0)
      setattr(combined, attr, d)

    # Non-concat per-side metadata (same across motions)
    combined.joint_names = singles[0].joint_names
    combined.tip_indices = singles[0].tip_indices

    # Euler vel computed later by the command term (needs env step_dt)
    combined.obj_euler_vel = {}

    # Multi-motion metadata
    n = len(singles)
    frames = torch.tensor([m.time_step_total for m in singles], dtype=torch.long, device=device)
    starts = torch.zeros(n, dtype=torch.long, device=device)
    starts[1:] = frames[:-1].cumsum(0)
    combined.num_motions = n
    combined.motion_num_frames = frames
    combined.length_starts = starts

    return combined


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

    if isinstance(cfg.motion_file, list):
      self.motion = ManipTransMotionData.from_multiple(
        cfg.motion_file, cfg.sides, n_hand_dofs, device=self.device
      )
    else:
      self.motion = ManipTransMotionData(
        cfg.motion_file, cfg.sides, n_hand_dofs, device=self.device
      )

    # Per-env motion assignment (for multi-motion). Initialized to 0 for
    # single-motion backward compatibility.
    self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # Populate per-side hinge-joint velocity trajectories now that env step
    # dt is known. Used at reset to give actuated objects a non-cold-start
    # initial velocity. Stored on `self.motion` for symmetry with obj_euler.
    step_dt = env.step_dt
    for side in cfg.sides:
      if side in self.motion.obj_euler:
        euler_np = self.motion.obj_euler[side].cpu().numpy()
        euler_vel = np.gradient(euler_np, axis=0) / step_dt  # (T, 3) rad/s
        self.motion.obj_euler_vel[side] = torch.tensor(
          euler_vel, dtype=torch.float32, device=self.device
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

    # All tracked bodies (17 per side, matching ManipTrans body_names minus wrist).
    # Each maps to a MANO joint for delta obs and tracking rewards.
    # ManipTrans xhand.py: hand2dex_mapping defines which robot bodies map to which MANO joints.
    _ALL_BODIES_MANO = (
      ("hand_thumb_bend_link", "thumb_proximal"),
      ("hand_thumb_rota_link1", "thumb_proximal"),
      ("hand_thumb_rota_link2", "thumb_intermediate"),
      ("hand_index_bend_link", "index_proximal"),
      ("hand_index_rota_link1", "index_proximal"),
      ("hand_index_rota_link2", "index_intermediate"),
      ("hand_mid_link1", "middle_proximal"),
      ("hand_mid_link2", "middle_intermediate"),
      ("hand_ring_link1", "ring_proximal"),
      ("hand_ring_link2", "ring_intermediate"),
      ("hand_pinky_link1", "pinky_proximal"),
      ("hand_pinky_link2", "pinky_intermediate"),
    )

    # Level 1/2 for rewards (one body per finger per level, same as before)
    _LEVEL1_BODIES = {
      "thumb": "hand_thumb_bend_link",
      "index": "hand_index_bend_link",
      "middle": "hand_mid_link1",
      "ring": "hand_ring_link1",
      "pinky": "hand_pinky_link1",
    }
    _LEVEL2_BODIES = {
      "thumb": "hand_thumb_rota_link2",
      "index": "hand_index_rota_link2",
      "middle": "hand_mid_link2",
      "ring": "hand_ring_link2",
      "pinky": "hand_pinky_link2",
    }
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
    wrist_ids, wrist_names = self.robot.find_joints_by_actuator_names((".*forearm.*",))
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

    # Time stepping
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # Adaptive sampling state — per-trajectory bins so multi-ref doesn't
    # cross-contaminate failure statistics between different motions.
    max_motion_frames = int(self.motion.motion_num_frames.max().item()) if self.motion.num_motions > 1 else self.motion.time_step_total
    self.bin_count = int(
      max_motion_frames // (1 / env.step_dt)
    ) + 1
    n_motions = max(self.motion.num_motions, 1)
    self.bin_failed_count = torch.zeros(
      n_motions, self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      n_motions, self.bin_count, dtype=torch.float, device=self.device
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

  # --- Flat index for multi-motion dispatch ---

  @property
  def _flat_idx(self) -> torch.Tensor:
    """Per-env flat frame index into the concatenated motion tensors.

    ``length_starts[motion_ids[i]] + time_steps[i]`` for each env i.
    For single-motion (num_motions=1), degenerates to ``time_steps``.
    """
    return self.motion.length_starts[self.motion_ids] + self.time_steps

  # --- Command property ---

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.ref_joint_pos, self.ref_joint_vel], dim=1)

  # --- Reference joint state (for reset) ---

  @property
  def ref_joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self._flat_idx]

  @property
  def ref_joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self._flat_idx]

  # --- MANO tracking targets ---

  @property
  def mano_wrist_pos_w(self) -> torch.Tensor:
    """MANO wrist positions. Shape: (B, n_sides, 3)."""
    parts = []
    for side in self._side_list:
      pos = self.motion.wrist_pos[side][self._flat_idx]  # (B, 3)
      pos = pos + self._env.scene.env_origins
      parts.append(pos)
    return torch.stack(parts, dim=1)

  @property
  def mano_wrist_rot_w(self) -> torch.Tensor:
    """MANO wrist rotation matrices. Shape: (B, n_sides, 3, 3)."""
    parts = []
    for side in self._side_list:
      parts.append(self.motion.wrist_rot[side][self._flat_idx])
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
      parts.append(self.motion.wrist_vel[side][self._flat_idx])
    return torch.stack(parts, dim=1)

  @property
  def mano_wrist_angvel_w(self) -> torch.Tensor:
    """MANO wrist angular velocities. Shape: (B, n_sides, 3)."""
    parts = []
    for side in self._side_list:
      parts.append(self.motion.wrist_angvel[side][self._flat_idx])
    return torch.stack(parts, dim=1)

  @property
  def mano_tip_pos_w(self) -> torch.Tensor:
    """MANO fingertip positions (5 per side). Shape: (B, n_sides, 5, 3)."""
    parts = []
    for side in self._side_list:
      all_joints = self.motion.joints[side][self._flat_idx]  # (B, 20, 3)
      tips = all_joints[:, self.motion.tip_indices[side]]  # (B, 5, 3)
      tips = tips + self._env.scene.env_origins[:, None, :]
      parts.append(tips)
    return torch.stack(parts, dim=1)

  @property
  def mano_tip_vel_w(self) -> torch.Tensor:
    """MANO fingertip velocities (5 per side). Shape: (B, n_sides, 5, 3)."""
    parts = []
    for side in self._side_list:
      all_vel = self.motion.joints_vel[side][self._flat_idx]  # (B, 20, 3)
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
      local_pts = self.motion.contact_pos_full[side][self._flat_idx]  # (B, 5, 3)
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
      parts.append(self.motion.contact_flags[side][self._flat_idx])
    return torch.stack(parts, dim=1)

  @property
  def mano_tips_distance(self) -> torch.Tensor:
    """Precomputed MANO tip-to-object-surface distance. Shape: (B, n_sides, 5)."""
    parts = []
    for side in self._side_list:
      parts.append(self.motion.tips_distance[side][self._flat_idx])
    return torch.stack(parts, dim=1)

  # --- All body joint tracking (17 per side: 12 non-tip + 5 tip) ---

  def mano_all_joints_pos_w(self, side: str) -> torch.Tensor:
    """MANO positions for all 12 non-tip joints. Shape: (B, 12, 3)."""
    mano_indices = self._all_mano_indices[side]
    all_joints = self.motion.joints[side][self._flat_idx]  # (B, 20, 3)
    pts = all_joints[:, mano_indices]  # (B, 12, 3)
    return pts + self._env.scene.env_origins[:, None, :]

  def robot_all_joints_pos_w(self, side: str) -> torch.Tensor:
    """Robot positions for all 12 non-tip bodies. Shape: (B, 12, 3)."""
    body_indices = self._all_body_indices[side]
    return self.robot.data.body_link_pos_w[:, body_indices]  # (B, 12, 3)

  def mano_all_joints_vel_w(self, side: str) -> torch.Tensor:
    """MANO velocities for all 12 non-tip joints. Shape: (B, 12, 3)."""
    mano_indices = self._all_mano_indices[side]
    all_vel = self.motion.joints_vel[side][self._flat_idx]  # (B, 20, 3)
    return all_vel[:, mano_indices]  # (B, 12, 3)

  def robot_all_joints_vel_w(self, side: str) -> torch.Tensor:
    """Robot velocities for all 12 non-tip bodies. Shape: (B, 12, 3)."""
    body_indices = self._all_body_indices[side]
    return self.robot.data.body_link_lin_vel_w[:, body_indices]  # (B, 12, 3)

  # --- Level 1/2 joint tracking ---

  def mano_level_pos_w(self, side: str, level: int) -> torch.Tensor:
    """MANO joint positions for level 1 or 2. Shape: (B, 5, 3)."""
    mano_indices = self._level1_mano_indices[side] if level == 1 else self._level2_mano_indices[side]
    all_joints = self.motion.joints[side][self._flat_idx]  # (B, 20, 3)
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
        pos = self.motion.obj_pos[side][self._flat_idx]
        pos = pos + self._env.scene.env_origins
        parts.append(pos)
    return torch.stack(parts, dim=1)

  @property
  def ref_obj_rotmat_w(self) -> torch.Tensor:
    """Reference object rotation matrices. Shape: (B, n_sides, 3, 3)."""
    parts = []
    for side in self._side_list:
      if side in self.motion.obj_rotmat:
        parts.append(self.motion.obj_rotmat[side][self._flat_idx])
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
        parts.append(self.motion.obj_vel[side][self._flat_idx])
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
        parts.append(self.motion.obj_angvel[side][self._flat_idx])
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
    """Next frame index, clamped to motion length."""
    return torch.clamp(self.time_steps + 1, max=self.motion.time_step_total - 1)

  @property
  def next_obj_pos_w(self) -> torch.Tensor:
    """Next-frame object positions. Shape: (B, n_sides, 3)."""
    next_t = self._next_time_steps()
    parts = []
    for side in self._side_list:
      if side in self.motion.obj_pos:
        pos = self.motion.obj_pos[side][next_t] + self._env.scene.env_origins
        parts.append(pos)
    return torch.stack(parts, dim=1)

  @property
  def next_obj_quat_w(self) -> torch.Tensor:
    """Next-frame object quaternions. Shape: (B, n_sides, 4)."""
    next_t = self._next_time_steps()
    parts = []
    for side in self._side_list:
      if side in self.motion.obj_rotmat:
        rotmat = self.motion.obj_rotmat[side][next_t]
        parts.append(quat_from_matrix(rotmat))
    return torch.stack(parts, dim=1)

  @property
  def next_obj_vel_w(self) -> torch.Tensor:
    """Next-frame object velocities. Shape: (B, n_sides, 3)."""
    next_t = self._next_time_steps()
    parts = []
    for side in self._side_list:
      if side in self.motion.obj_vel:
        parts.append(self.motion.obj_vel[side][next_t])
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

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Multi-motion: sample a new motion_id per env at each reset
    if self.motion.num_motions > 1:
      self.motion_ids[env_ids] = torch.randint(
        0, self.motion.num_motions, (len(env_ids),), device=self.device
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
          obj_angvel = self.motion.obj_angvel[side][self._flat_idx[env_ids]]
          root_state = torch.cat([obj_pos, obj_quat, obj_vel, obj_angvel], dim=-1)
          obj.write_root_state_to_sim(root_state, env_ids=env_ids)
          obj.reset(env_ids=env_ids)
        elif pin_mode == "actuated":
          # Local-frame (mocap_base child) joint positions. Use precomputed
          # unwrapped intrinsic-XYZ Euler to avoid the ±π canonicalization
          # jump that would send the actuator target spinning 360° between
          # frames (the "light object rotates" bug).
          t = self._flat_idx[env_ids]
          local_pos = self.motion.obj_pos[side][t]       # (N, 3)
          euler = self.motion.obj_euler[side][t]         # (N, 3)
          joint_pos = torch.cat([local_pos, euler], dim=-1)  # (N, 6)
          # Initial joint velocities to match the reference trajectory (no
          # cold start from rest):
          #   slide qvel = world linear velocity (slide axes are world-aligned)
          #   hinge qvel = d/dt of unwrapped Euler (precomputed)
          lin_vel = self.motion.obj_vel[side][t]         # (N, 3)
          euler_vel = self.motion.obj_euler_vel[side][t]  # (N, 3)
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

    # Wrap around envs that exceeded their motion's length (per-motion for
    # multi-motion, global time_step_total for single-motion).
    per_env_max = self.motion.motion_num_frames[self.motion_ids]
    wrap_ids = torch.where(self.time_steps >= per_env_max)[0]
    if wrap_ids.numel() > 0:
      self._resample_command(wrap_ids)

    # Pin objects to reference trajectory every step
    if self.cfg.pin_objects and self.has_objects:
      pin_mode = getattr(self.cfg, "pin_mode", "hard")
      if pin_mode == "hard":
        # Fixed temporal cadence: pin every `pin_interval` env steps. T=1 is
        # original full hard pin; T>1 lets the object drift T-1 steps per
        # cycle. Skip reset step 0 — _resample_command already placed the
        # object.
        T = max(1, int(self.cfg.pin_interval))
        ep_len = self._env.episode_length_buf
        pin_mask = (ep_len > 0) & ((ep_len % T) == 0)
        pin_ids = torch.where(pin_mask)[0]
        if pin_ids.numel() > 0:
          for side in self._side_list:
            if side not in self.motion.obj_pos:
              continue
            obj_name = self.cfg.object_entity_names[side]
            obj: Entity = self._env.scene[obj_name]
            si = self._side_list.index(side)
            obj_pos = self.ref_obj_pos_w[pin_ids, si]
            obj_quat = self.ref_obj_quat_w[pin_ids, si]
            obj_vel = self.motion.obj_vel[side][self._flat_idx[pin_ids]]
            obj_angvel = self.motion.obj_angvel[side][self._flat_idx[pin_ids]]
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
          local_pos = self.motion.obj_pos[side][self._flat_idx]  # (B, 3)
          euler = self.motion.obj_euler[side][self._flat_idx]  # (B, 3)
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
          ref_vel = self.motion.obj_vel[side][self._flat_idx]
          ref_angvel = self.motion.obj_angvel[side][self._flat_idx]
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
    episode_failed = self._env.termination_manager.terminated[env_ids]
    env_motion_ids = self.motion_ids[env_ids]

    if torch.any(episode_failed):
      per_env_max = self.motion.motion_num_frames[env_motion_ids].float()
      current_bin_index = torch.clamp(
        (self.time_steps[env_ids].float() * self.bin_count / per_env_max).long(),
        0,
        self.bin_count - 1,
      )
      failed_ids = env_motion_ids[episode_failed]
      failed_bins = current_bin_index[episode_failed]
      self._current_bin_failed.zero_()
      for mid in range(self.motion.num_motions):
        mask = failed_ids == mid
        if mask.any():
          self._current_bin_failed[mid] = torch.bincount(
            failed_bins[mask], minlength=self.bin_count
          ).float()

    # Per-env: sample from the assigned motion's failure distribution
    per_env_max = self.motion.motion_num_frames[env_motion_ids].float()
    sampled_times = torch.zeros(len(env_ids), device=self.device)
    for mid in range(self.motion.num_motions):
      mask = env_motion_ids == mid
      if not mask.any():
        continue
      probs = (
        self.bin_failed_count[mid]
        + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
      )
      probs = torch.nn.functional.pad(
        probs.unsqueeze(0).unsqueeze(0),
        (0, self.cfg.adaptive_kernel_size - 1),
        mode="replicate",
      )
      probs = torch.nn.functional.conv1d(
        probs, self.kernel.view(1, 1, -1)
      ).view(-1)
      probs = probs / probs.sum()
      n = mask.sum().item()
      bins = torch.multinomial(probs, n, replacement=True)
      motion_frames = self.motion.motion_num_frames[mid].float()
      sampled_times[mask] = (
        (bins.float() + torch.rand(n, device=self.device))
        / self.bin_count * (motion_frames - 1)
      )
    self.time_steps[env_ids] = sampled_times.long()

  def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
    # ManipTrans: floor(seq_len * 0.99 * rand) — uniform over 99% of per-motion length
    per_env_max = self.motion.motion_num_frames[self.motion_ids[env_ids]].float()
    self.time_steps[env_ids] = (
      torch.rand(len(env_ids), device=self.device) * 0.99 * per_env_max
    ).long()


@dataclass(kw_only=True)
class ManipTransCommandCfg(CommandTermCfg):
  """Configuration for ManipTrans motion command."""

  motion_file: str | list[str]
  """Single motion.npz path, or a list of paths for multi-trajectory training.
  When a list, ``ManipTransMotionData.from_multiple()`` concatenates all motions
  into one dataset with per-motion offset indexing, and each env samples a random
  motion at each reset."""
  entity_name: str
  sides: tuple[str, ...]
  wrist_body_names: dict[str, str]
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
  N-1 steps between snaps. T >= episode_length is effectively no pin.
  """
  joint_position_range: tuple[float, float] = (0.0, 0.0)
  sampling_mode: Literal["adaptive", "uniform", "start"] = "uniform"
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001

  def build(self, env: ManagerBasedRlEnv) -> ManipTransCommand:
    return ManipTransCommand(self, env)
