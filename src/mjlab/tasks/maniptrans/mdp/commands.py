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
from mjlab.sensor import ContactSensor
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


def _bake_object_sdf_grid(
  mesh_path: str,
  mesh_scale: float,
  extent: float,
  N: int,
  device: str,
) -> torch.Tensor:
  """Bake a signed-distance + gradient grid for an object mesh.

  Output layout: ``(4, N, N, N)`` float32 — channel 0 is the signed distance
  (positive outside, negative inside; standard SDF convention), channels 1-3
  are the analytic gradient via central differences along x/y/z. Grid spans
  ``[-extent, extent]^3`` in the object-local frame; voxel side is
  ``2 * extent / (N - 1)`` (corner-aligned, F.grid_sample-friendly).

  Caches the baked grid under ``<mesh_dir>/.sdf_cache/sdf_e{extent}_n{N}_s{scale}.pt``
  so subsequent runs on the same object load instantly. Cache key encodes
  the parameters that change the grid contents; mesh mtime is *not* hashed
  (regenerate the cache by deleting the file if you replace the .obj).
  """
  import hashlib
  from pathlib import Path

  import trimesh

  mesh_path_p = Path(mesh_path)
  cache_dir = mesh_path_p.parent / ".sdf_cache"
  # Cache key encodes the bake-time parameters that change the grid contents,
  # plus a "method" tag so old caches with the np.gradient-based gradient are
  # silently abandoned in favour of the analytic-normal v2 layout.
  key = hashlib.md5(
    f"{mesh_path_p.name}|method=v2_analytic|e={extent}|n={N}|s={mesh_scale}".encode()
  ).hexdigest()[:12]
  cache_file = cache_dir / f"sdf_v2_e{extent}_n{N}_s{mesh_scale}_{key}.pt"

  if cache_file.exists():
    return torch.load(cache_file, map_location=device, weights_only=True).to(
      device=device, dtype=torch.float32
    )

  mesh = trimesh.load(mesh_path_p, process=False, force="mesh")
  if not isinstance(mesh, trimesh.Trimesh):
    raise TypeError(
      f"SDF bake expected a single trimesh.Trimesh, got {type(mesh)} from {mesh_path!r}"
    )
  if mesh_scale != 1.0:
    mesh.apply_scale(mesh_scale)

  xs = np.linspace(-extent, extent, N, dtype=np.float32)
  X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
  pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

  # closest_point gives both unsigned distance and the projection onto the
  # surface (`closest`). contains gives the sign. Together: standard SDF
  # (positive outside, negative inside) plus analytic outward normals via
  # the displacement (pts - closest) / |pts - closest|. The analytic
  # normal is unit-length by construction near the surface; np.gradient
  # over a discrete grid systematically under-magnitudes the gradient by
  # ~25% at the surface, which makes the "direction" signal noisy.
  closest, dist, _ = trimesh.proximity.closest_point(mesh, pts)
  inside = mesh.contains(pts)
  sd = np.where(inside, -dist, dist).astype(np.float32)

  # Analytic SDF gradient: ∇sdf points in the direction of increasing SDF.
  # - outside (sd > 0): away from the surface, i.e. (pts - closest) / |...|.
  # - inside  (sd < 0): toward the surface,    i.e. (closest - pts) / |...|.
  # Combined: sign(sd) * (pts - closest) / |pts - closest|, with sign +1
  # outside and -1 inside.
  sign = np.where(inside, -1.0, 1.0).astype(np.float32)
  disp = (pts - closest).astype(np.float32)
  unit = disp / np.maximum(
    np.linalg.norm(disp, axis=-1, keepdims=True), 1e-8
  )
  grad_pts = (sign[:, None] * unit).astype(np.float32)  # (M, 3)
  grad = grad_pts.reshape(N, N, N, 3).transpose(3, 0, 1, 2)  # (3, N, N, N)
  sdf = sd.reshape(N, N, N)
  grid = np.concatenate([sdf[None], grad], axis=0)  # (4, N, N, N)

  cache_dir.mkdir(parents=True, exist_ok=True)
  torch.save(torch.from_numpy(grid), cache_file)
  return torch.from_numpy(grid).to(device=device, dtype=torch.float32)


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
    motion_index: int | None = None,
  ) -> None:
    if (
      isinstance(motion_file, (str, bytes))
      and str(motion_file).endswith(".pt")
    ):
      datas, motion_files, packed_meta = self._load_packed_pt(
        str(motion_file), motion_index
      )
    else:
      if motion_index is not None:
        raise ValueError(
          "motion_index only applies to packed .pt motion files; got "
          f"{motion_file!r}"
        )
      motion_files = (
        [motion_file] if isinstance(motion_file, (str, bytes)) else list(motion_file)
      )
      if len(motion_files) == 0:
        raise ValueError("motion_file must be a non-empty path or list of paths")
      datas = [dict(np.load(p, allow_pickle=True)) for p in motion_files]
      packed_meta = None
    self.packed_meta: dict | None = packed_meta
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
  def _load_packed_pt(
    path: str, motion_index: int | None
  ) -> tuple[list[dict[str, np.ndarray]], list[str], dict]:
    """Load a packed motion .pt file produced by package_motion_batch.py.

    Returns (datas, motion_files, packed_meta) where datas is a list of
    dict-of-numpy-arrays mimicking np.load() output for each selected motion,
    motion_files is a list of synthetic per-motion identifiers (for error
    messages), and packed_meta carries embedded task_info-equivalent fields
    (right/left_object_mesh_dir, scales, pool_rel_dir).
    """
    packed = torch.load(path, weights_only=False)
    motion_num_frames = packed["motion_num_frames"].tolist()
    length_starts = packed["length_starts"].tolist()
    motion_filenames = packed.get("motion_filename", [])
    M_total = len(motion_num_frames)

    if motion_index is None or motion_index < 0:
      slice_idxs = list(range(M_total))
    else:
      slice_idxs = [motion_index]

    total_frames = int(sum(motion_num_frames))
    flat_keys = []
    for k, v in packed.items():
      if torch.is_tensor(v) and v.ndim >= 1 and v.size(0) == total_frames:
        flat_keys.append(k)

    datas: list[dict[str, np.ndarray]] = []
    motion_files: list[str] = []
    for m in slice_idxs:
      s = length_starts[m]
      T_m = motion_num_frames[m]
      d: dict[str, np.ndarray] = {}
      for k in flat_keys:
        d[k] = packed[k][s : s + T_m].cpu().numpy()
      for jn_key in ("mano_right_joint_names", "mano_left_joint_names"):
        if jn_key in packed:
          d[jn_key] = np.array(packed[jn_key])
      datas.append(d)
      ident = motion_filenames[m] if m < len(motion_filenames) else f"motion[{m}]"
      motion_files.append(f"{path}#{ident}")

    packed_meta = {
      "right_object_mesh_dir": packed.get("right_object_mesh_dir"),
      "left_object_mesh_dir": packed.get("left_object_mesh_dir"),
      "right_object_mesh_scale": packed.get("right_object_mesh_scale"),
      "left_object_mesh_scale": packed.get("left_object_mesh_scale"),
      "pool_rel_dir": packed.get("pool_rel_dir"),
      "object_id": packed.get("object_id"),
    }
    return datas, motion_files, packed_meta

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
      cfg.motion_file, cfg.sides, n_hand_dofs, device=self.device,
      motion_index=cfg.motion_index,
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

    # Per-side: palm site (canonical Z-up wrist frame) and tip / contact sites.
    self._side_list = list(cfg.sides)
    self._palm_site_indices: dict[str, int] = {}
    self._tip_site_indices: dict[str, list[int]] = {}
    self._contact_site_indices: dict[str, list[int]] = {}
    for side in cfg.sides:
      self._palm_site_indices[side] = self.robot.site_names.index(f"{side}_palm")
      tip_names = [f"track_hand_{side}_{finger}_tip" for finger in FINGER_NAMES]
      ids, _ = self.robot.find_sites(tip_names, preserve_order=True)
      self._tip_site_indices[side] = ids
      contact_names = [f"contact_{side}_{finger}_tip" for finger in FINGER_NAMES]
      contact_ids, _ = self.robot.find_sites(contact_names, preserve_order=True)
      self._contact_site_indices[side] = contact_ids

    # Per-side object SDF grids (4, N, N, N): channel 0 = signed distance
    # (positive outside), channels 1-3 = ∇sdf via central differences. Baked
    # once at command init from the visual mesh; cached on disk under the
    # mesh dir so subsequent runs load instantly. Used by the `sdf_query`
    # helper for SDF-based observations and rewards.
    self._obj_sdf_grids: dict[str, torch.Tensor] = {}
    self._obj_sdf_extent: float = float(cfg.obj_sdf_grid_extent)
    self._obj_sdf_n: int = int(cfg.obj_sdf_grid_n)
    if cfg.object_mesh_paths is not None:
      scales = cfg.object_mesh_scales or {}
      for side, mesh_path in cfg.object_mesh_paths.items():
        if side not in cfg.sides:
          continue
        if mesh_path is None:
          continue
        scale = float(scales.get(side, 1.0))
        self._obj_sdf_grids[side] = _bake_object_sdf_grid(
          mesh_path, scale, self._obj_sdf_extent, self._obj_sdf_n,
          device=str(self.device),
        )

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

    # Per-side / per-finger tracking metrics — 1-to-1 mirror of the
    # tracking-reward terms registered in
    # `config/base.py:_add_per_side_rewards`.
    side_prefixes = {"right": "r", "left": "l"}
    for side in cfg.sides:
      p = side_prefixes[side]
      self.metrics[f"error_wrist_pos_{p}"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics[f"error_wrist_rot_{p}"] = torch.zeros(self.num_envs, device=self.device)
      for finger in FINGER_NAMES:
        self.metrics[f"error_tip_pos_{p}_{finger}"] = torch.zeros(self.num_envs, device=self.device)
      for level in (1, 2):
        self.metrics[f"error_level{level}_{p}"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics[f"error_wrist_vel_{p}"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics[f"error_wrist_angvel_{p}"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics[f"error_joints_vel_{p}"] = torch.zeros(self.num_envs, device=self.device)
      if self.has_objects:
        self.metrics[f"error_obj_pos_{p}"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics[f"error_obj_rot_{p}"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics[f"error_obj_vel_{p}"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics[f"error_obj_angvel_{p}"] = torch.zeros(self.num_envs, device=self.device)

    # Conditional contact metrics. Two gates:
    #  - count_ref: Σ ref_flag                       (denominator for ref_dist + found)
    #  - count_contact: Σ ref_flag · found           (denominator for pen, force)
    # pen/force averages exclude frames where the policy missed contact — otherwise
    # zero-force misses pull the means toward 0 and obscure in-contact behavior.
    # found = count_contact / count_ref = hit rate (no separate sum tensor needed).
    # Emitted in reset() as count-weighted means across resetting envs.
    self._contact_sum: dict[str, torch.Tensor] = {}
    self._contact_count_ref: dict[str, torch.Tensor] = {}
    self._contact_count_contact: dict[str, torch.Tensor] = {}
    if self.has_objects:
      for side in cfg.sides:
        p = side_prefixes[side]
        for finger in FINGER_NAMES:
          k = f"{p}_{finger}"
          self._contact_count_ref[k]     = torch.zeros(self.num_envs, device=self.device)
          self._contact_count_contact[k] = torch.zeros(self.num_envs, device=self.device)
          for name in ("ref_dist", "pen", "force"):
            self._contact_sum[f"contact_{name}_{k}"] = torch.zeros(
              self.num_envs, device=self.device
            )

    # Per-(side, finger) consecutive-miss counter:
    #   miss = ref_flag==1 AND found==0   →  counter += 1
    #   else (ref_flag==0 OR found==1)    →  counter = 0
    # Used by the contact_missed_too_long termination AND tracked as a
    # per-finger episode-max metric so we can pick a sensible threshold.
    self.contact_miss_counter = torch.zeros(
      self.num_envs, len(self._side_list), len(FINGER_NAMES), device=self.device
    )
    self.contact_miss_max = torch.zeros_like(self.contact_miss_counter)

    # Per-step pin-fired buffer — used by the pin physics path in `_apply` and
    # by the `pin_penalty` reward (registered by `add_object_interaction_rewards`).
    # Always allocated so reads during hand-only mode (where it stays zero) work.
    self.pin_fired_this_step = torch.zeros(
      self.num_envs, len(self._side_list), dtype=torch.bool, device=self.device
    )

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
    """Robot palm site positions (canonical Z-up). Shape: (B, n_sides, 3)."""
    parts = []
    for side in self._side_list:
      idx = self._palm_site_indices[side]
      parts.append(self.robot.data.site_pos_w[:, idx])
    return torch.stack(parts, dim=1)

  @property
  def robot_wrist_quat_w(self) -> torch.Tensor:
    """Robot palm site quaternions (canonical Z-up). Shape: (B, n_sides, 4)."""
    parts = []
    for side in self._side_list:
      idx = self._palm_site_indices[side]
      parts.append(self.robot.data.site_quat_w[:, idx])
    return torch.stack(parts, dim=1)

  @property
  def robot_wrist_rot_w(self) -> torch.Tensor:
    """Robot wrist rotation matrices (from quat). Shape: (B, n_sides, 3, 3)."""
    return matrix_from_quat(self.robot_wrist_quat_w)

  @property
  def robot_wrist_vel_w(self) -> torch.Tensor:
    """Robot palm site linear velocities. Shape: (B, n_sides, 3)."""
    parts = []
    for side in self._side_list:
      idx = self._palm_site_indices[side]
      parts.append(self.robot.data.site_lin_vel_w[:, idx])
    return torch.stack(parts, dim=1)

  @property
  def robot_wrist_angvel_w(self) -> torch.Tensor:
    """Robot palm site angular velocities. Shape: (B, n_sides, 3)."""
    parts = []
    for side in self._side_list:
      idx = self._palm_site_indices[side]
      parts.append(self.robot.data.site_ang_vel_w[:, idx])
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

  def sdf_query(
    self, world_pts: torch.Tensor, side: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Query the per-side baked SDF + gradient at arbitrary world-frame points.

    Args:
      world_pts: ``(B, K, 3)`` (or ``(B, 3)``) world-frame query points.
      side: which side's SDF to use (must be in ``cfg.sides`` and have a
        baked grid; raises KeyError otherwise).

    Returns:
      ``(sdf, grad_world)`` of shapes ``(B, K)`` and ``(B, K, 3)`` (or
      ``(B,)`` and ``(B, 3)`` if input is 2-D). ``sdf`` is the signed
      distance in metres (positive outside, negative inside, standard
      convention). ``grad_world`` is the SDF gradient expressed in world
      frame, magnitude ≈ 1 near the surface (it's an approximate surface
      normal for outside-points).

    Implementation: transforms the query points into the object's local
    frame using ``sim_obj_pos_w`` and ``sim_obj_quat_w``, normalizes to
    grid coords in ``[-1, 1]``, samples the baked grid via
    ``F.grid_sample`` (trilinear, border padding) in a single call. The
    gradient is sampled in the same call (channels 1-3 of the grid) and
    rotated back to world frame. Outside the grid box, the returned SDF /
    gradient is the value at the nearest box face — so callers should
    ensure ``cfg.obj_sdf_grid_extent`` is large enough to enclose the
    expected query region (default 0.30 m → 60 cm cube around the object).
    """
    if side not in self._obj_sdf_grids:
      raise KeyError(
        f"sdf_query: no SDF grid baked for side {side!r}. Set "
        f"`object_mesh_paths[{side!r}]` on the ManipTransCommandCfg."
      )
    grid = self._obj_sdf_grids[side]  # (4, N, N, N)
    si = self._side_list.index(side)
    obj_pos = self.sim_obj_pos_w[:, si]  # (B, 3)
    obj_quat = self.sim_obj_quat_w[:, si]  # (B, 4)

    squeeze_K = world_pts.dim() == 2
    if squeeze_K:
      world_pts = world_pts.unsqueeze(1)  # (B, 1, 3)

    B, K = world_pts.shape[0], world_pts.shape[1]
    # World → object-local: local = R(q_obj)^-1 (world - obj_pos).
    # Broadcast quat over K via the lab_api helper (works on (..., 4) / (..., 3)).
    delta = world_pts - obj_pos[:, None, :]  # (B, K, 3)
    quat_b = obj_quat[:, None, :].expand(B, K, 4)
    from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
    local = quat_apply_inverse(quat_b, delta)  # (B, K, 3) in object frame

    # Normalize to F.grid_sample coords in [-1, 1].
    norm = local / self._obj_sdf_extent  # (B, K, 3)
    # F.grid_sample 5-D: input (N, C, D, H, W); grid last-axis is (x, y, z)
    # mapping to (W, H, D) of the input. Our grid is laid out so dim-order
    # is (x_idx, y_idx, z_idx) -- i.e., D=x, H=y, W=z. So the grid sample's
    # "(x, y, z)" must be passed as (z_norm, y_norm, x_norm) to align.
    norm_zyx = norm.flip(-1)  # (B, K, 3) reordered to (z, y, x)
    sample_grid = norm_zyx.reshape(1, B * K, 1, 1, 3)
    grid_5d = grid.unsqueeze(0)  # (1, 4, N, N, N)
    out = torch.nn.functional.grid_sample(
      grid_5d, sample_grid, mode="bilinear",
      padding_mode="border", align_corners=True,
    )  # (1, 4, B*K, 1, 1)
    out = out.view(4, B, K).permute(1, 2, 0)  # (B, K, 4)
    sdf = out[..., 0]  # (B, K)
    # Channels 1-3 were baked as analytic outward normals (unit length at
    # each grid point). Trilinear interpolation between neighbouring voxels
    # with slightly-different normals (mesh curvature) produces a vector
    # that's *near* unit length but typically 0.7-0.9 in practice. The
    # actor needs a direction signal not a magnitude (magnitude info lives
    # in the SDF channel), so re-normalize here to a clean unit vector.
    grad_local = out[..., 1:]  # (B, K, 3) in object frame
    grad_local = grad_local / (
      grad_local.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    )
    grad_world = quat_apply(quat_b, grad_local)  # (B, K, 3)

    if squeeze_K:
      sdf = sdf.squeeze(1)
      grad_world = grad_world.squeeze(1)
    return sdf, grad_world

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
    side_prefixes = {"right": "r", "left": "l"}
    for si, side in enumerate(self._side_list):
      p = side_prefixes[side]

      # Wrist position (m)
      self.metrics[f"error_wrist_pos_{p}"] = torch.norm(
        self.mano_wrist_pos_w[:, si] - self.robot_wrist_pos_w[:, si], dim=-1
      )

      # Wrist rotation (rad)
      self.metrics[f"error_wrist_rot_{p}"] = quat_error_magnitude(
        self.mano_wrist_quat_w[:, si], self.robot_wrist_quat_w[:, si]
      )

      # Per-finger fingertip position (m)
      tip_err = torch.norm(
        self.mano_tip_pos_w[:, si] - self.robot_tip_pos_w[:, si], dim=-1
      )  # (B, 5)
      for fi, finger in enumerate(FINGER_NAMES):
        self.metrics[f"error_tip_pos_{p}_{finger}"] = tip_err[:, fi]

      # Level 1, 2 joint position (m, mean over 5 joints)
      for level in (1, 2):
        mano_pos = self.mano_level_pos_w(side, level)   # (B, 5, 3)
        robot_pos = self.robot_level_pos_w(side, level)  # (B, 5, 3)
        self.metrics[f"error_level{level}_{p}"] = torch.norm(
          mano_pos - robot_pos, dim=-1
        ).mean(dim=-1)

      # Wrist linear / angular velocity (m/s, rad/s; mean over xyz)
      self.metrics[f"error_wrist_vel_{p}"] = torch.mean(
        torch.abs(self.mano_wrist_vel_w[:, si] - self.robot_wrist_vel_w[:, si]),
        dim=-1,
      )
      self.metrics[f"error_wrist_angvel_{p}"] = torch.mean(
        torch.abs(self.mano_wrist_angvel_w[:, si] - self.robot_wrist_angvel_w[:, si]),
        dim=-1,
      )

      # All-17-bodies joint velocity (matches joints_vel_error_exp's calc)
      body_delta = (
        self.mano_all_joints_vel_w(side) - self.robot_all_joints_vel_w(side)
      )  # (B, 12, 3)
      tip_mano_vel = self.mano_tip_vel_w[:, si]                   # (B, 5, 3)
      tip_robot_vel = self.robot_all_joints_vel_w(side)[:, -5:]   # (B, 5, 3)
      all_delta = torch.cat([body_delta, tip_mano_vel - tip_robot_vel], dim=1)
      self.metrics[f"error_joints_vel_{p}"] = (
        all_delta.abs().mean(dim=-1).mean(dim=-1)
      )

      # Object tracking — mirror obj_{pos,rot,vel,angvel}_error_exp rewards.
      if self.has_objects:
        self.metrics[f"error_obj_pos_{p}"] = torch.norm(
          self.ref_obj_pos_w[:, si] - self.sim_obj_pos_w[:, si], dim=-1
        )
        self.metrics[f"error_obj_rot_{p}"] = quat_error_magnitude(
          self.ref_obj_quat_w[:, si], self.sim_obj_quat_w[:, si]
        )
        self.metrics[f"error_obj_vel_{p}"] = torch.mean(
          torch.abs(self.ref_obj_vel_w[:, si] - self.sim_obj_vel_w[:, si]),
          dim=-1,
        )
        self.metrics[f"error_obj_angvel_{p}"] = torch.mean(
          torch.abs(self.ref_obj_angvel_w[:, si] - self.sim_obj_angvel_w[:, si]),
          dim=-1,
        )

        # Per-finger contact telemetry. Two gates per (side, finger):
        #   - ref_flag==1            → ref_dist + found (hit rate)
        #   - ref_flag==1 AND found  → pen + force (in-contact behavior)
        pen_sensor: ContactSensor = self._env.scene[f"{p}_fingertip_penetration"]
        force_sensor: ContactSensor = self._env.scene[f"{p}_fingertip_contact"]
        for fi, finger in enumerate(FINGER_NAMES):
          k = f"{p}_{finger}"
          flag = self.ref_contact_flags[:, si, fi]
          found = (pen_sensor.data.found[:, fi] > 0).to(flag.dtype)
          contact_gate = flag * found
          ref_dist = torch.norm(
            self.ref_contact_pos_w[:, si, fi] - self.robot_tip_pos_w[:, si, fi],
            dim=-1,
          )
          pen = torch.clamp(-pen_sensor.data.dist[:, fi], min=0.0)
          force = torch.norm(force_sensor.data.force[:, fi], dim=-1)
          self._contact_sum[f"contact_ref_dist_{k}"] += ref_dist * flag
          self._contact_sum[f"contact_pen_{k}"]      += pen      * contact_gate
          self._contact_sum[f"contact_force_{k}"]    += force    * contact_gate
          self._contact_count_ref[k]     += flag
          self._contact_count_contact[k] += contact_gate
          # Consecutive-miss counter: increments on (flag==1 AND found==0),
          # resets on (flag==0 OR found==1). The single-line form below
          # captures both: miss=1 → (ctr+1)*1=ctr+1; miss=0 → (ctr+1)*0=0.
          miss = flag * (1.0 - found)
          self.contact_miss_counter[:, si, fi] = (
            self.contact_miss_counter[:, si, fi] + 1.0
          ) * miss
          self.contact_miss_max[:, si, fi] = torch.maximum(
            self.contact_miss_max[:, si, fi], self.contact_miss_counter[:, si, fi]
          )

  def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
    extras = super().reset(env_ids)
    if not self.has_objects:
      return extras
    side_prefixes = {"right": "r", "left": "l"}
    for si, side in enumerate(self._side_list):
      p = side_prefixes[side]
      for fi, finger in enumerate(FINGER_NAMES):
        k = f"{p}_{finger}"
        count_ref     = self._contact_count_ref[k]
        count_contact = self._contact_count_contact[k]
        total_ref     = count_ref[env_ids].sum().clamp(min=1.0)
        total_contact = count_contact[env_ids].sum().clamp(min=1.0)
        # ref_dist + found gated on ref_flag==1; pen + force gated on ref_flag==1 AND found.
        extras[f"contact_ref_dist_{k}"] = (
          self._contact_sum[f"contact_ref_dist_{k}"][env_ids].sum() / total_ref
        ).item()
        extras[f"contact_found_{k}"] = (count_contact[env_ids].sum() / total_ref).item()
        extras[f"contact_pen_{k}"] = (
          self._contact_sum[f"contact_pen_{k}"][env_ids].sum() / total_contact
        ).item()
        extras[f"contact_force_{k}"] = (
          self._contact_sum[f"contact_force_{k}"][env_ids].sum() / total_contact
        ).item()
        # Per-finger episode-max consecutive-miss streak (mean over resetting envs).
        extras[f"contact_miss_max_{k}"] = self.contact_miss_max[env_ids, si, fi].mean().item()
        for name in ("ref_dist", "pen", "force"):
          self._contact_sum[f"contact_{name}_{k}"][env_ids] = 0.0
        count_ref[env_ids]     = 0.0
        count_contact[env_ids] = 0.0
        self.contact_miss_counter[env_ids, si, fi] = 0.0
        self.contact_miss_max[env_ids, si, fi]     = 0.0
    return extras

  def _resample_command(self, env_ids: torch.Tensor) -> None:
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

    # Warm-start with ManipTrans-matched noise (training-time DR), scaled by
    # cfg.init_noise_scale. Default 1.0 = original ManipTrans behavior; set
    # 0.0 in eval / rollout for the deterministic retargeted ref pose with
    # no DR perturbation. (mjlab events fire BEFORE command_manager.reset,
    # so the noise CAN'T live as an EventTermCfg — _resample_command would
    # overwrite it with ref pose. Noise has to ride on top of ref pose
    # inside the same reset hook.)
    #
    # Scales at noise_scale=1.0:
    # - Wrist translation: randn * 0.01 (~1cm)
    # - Wrist rotation:    randn * pi/18 (~10°)
    # - Finger DOFs:       randn * (joint_range / 8)
    # - Wrist velocity:    randn * 0.01
    # - Finger velocity:   randn * 0.1 (REPLACES ref_joint_vel for fingers)
    joint_pos = self.ref_joint_pos[env_ids].clone()
    joint_vel = self.ref_joint_vel[env_ids].clone()

    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    noise_scale = float(getattr(self.cfg, "init_noise_scale", 1.0))

    if noise_scale > 0.0:
      joint_pos[:, self._wrist_trans_ids] += (
        torch.randn(len(env_ids), len(self._wrist_trans_ids), device=self.device)
        * 0.01 * noise_scale
      )
      joint_pos[:, self._wrist_rot_ids] += (
        torch.randn(len(env_ids), len(self._wrist_rot_ids), device=self.device)
        * (3.14159265 / 18.0) * noise_scale
      )
      finger_range = (
        soft_limits[:, self._finger_joint_ids, 1] - soft_limits[:, self._finger_joint_ids, 0]
      )
      joint_pos[:, self._finger_joint_ids] += (
        torch.randn(len(env_ids), len(self._finger_joint_ids), device=self.device)
        * (finger_range / 8.0) * noise_scale
      )
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])

    if noise_scale > 0.0:
      joint_vel[:, self._wrist_joint_ids] += (
        torch.randn(len(env_ids), len(self._wrist_joint_ids), device=self.device)
        * 0.01 * noise_scale
      )
      # Finger velocity REPLACES ref_joint_vel under noise (original
      # ManipTrans behavior). With noise_scale=0 we keep ref_joint_vel.
      joint_vel[:, self._finger_joint_ids] = (
        torch.randn(len(env_ids), len(self._finger_joint_ids), device=self.device)
        * 0.1 * noise_scale
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
          omega_rot = float(self.cfg.xfrc_omega_rot)
          if omega_rot > 0.0:
            # Anisotropic inertia-tensor mode. I_body = R(iquat)·diag(body_inertia)·R(iquat)^T
            # is the full inertia tensor in body frame (body_iquat maps principal→body
            # frame; identity if MuJoCo auto-aligned body frame to principal axes).
            # I_world = R_sim · I_body · R_sim^T per env per step.
            body_id = int(obj.indexing.body_ids[0])
            I_diag_p = self._env.sim.model.body_inertia[0, body_id].to(
              sim_quat.device, dtype=sim_quat.dtype
            )  # (3,) principal-frame diagonal
            iquat = self._env.sim.model.body_iquat[0, body_id].to(
              sim_quat.device, dtype=sim_quat.dtype
            )  # (4,) principal→body
            R_ip = matrix_from_quat(iquat.unsqueeze(0))[0]  # (3, 3)
            I_body = R_ip @ torch.diag(I_diag_p) @ R_ip.T  # (3, 3) full body-frame inertia
            R_sim = matrix_from_quat(sim_quat)  # (B, 3, 3)
            I_world = R_sim @ I_body @ R_sim.transpose(-1, -2)  # (B, 3, 3)
            zeta_rot = float(self.cfg.xfrc_zeta_rot)
            kp_rot_eff = omega_rot * omega_rot
            kv_rot_eff = 2.0 * zeta_rot * omega_rot
            ang_err = ref_angvel - sim_angvel
            torque = (
              kp_rot_eff * torch.einsum("bij,bj->bi", I_world, axis_angle)
              + kv_rot_eff * torch.einsum("bij,bj->bi", I_world, ang_err)
            )
          else:
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
  motion_index: "int | None" = None
  """For a single packed .pt motion_file, optionally select one motion (M=1).
  None = use all M motions in the packed file (multi-reference). No-op when
  motion_file is an .npz path or list of .npz paths."""
  entity_name: str
  sides: tuple[str, ...]
  body_mapping: dict
  """Per-hand body-name mapping. Keys: 'all' (sequence of (body, mano_joint)
  pairs for all tracked non-tip bodies), 'level1' and 'level2' (dicts from
  finger name to the body used for the L1/L2 tracking reward). Supplied by
  the hand's asset_zoo constants module (e.g. BODY_MAPPING in
  asset_zoo.hands.xhand.constants). Side prefix is applied at lookup time."""
  object_entity_names: dict[str, str] = None  # side → entity name, e.g. {"right": "object_right"}
  object_mesh_paths: dict[str, str] | None = None
  """Per-side absolute path to the visual mesh of each object (e.g. the
  object pool's `visual.obj`). Used to bake an object-local SDF grid at
  command init for SDF-based observations and rewards. None disables baking.
  """
  object_mesh_scales: dict[str, float] | None = None
  """Per-side mesh scale to apply at SDF bake time (matches the scene entity's
  spawn scale). Defaults to 1.0 per side if `object_mesh_paths` is set but
  scales are not. Ignored when `object_mesh_paths` is None.
  """
  obj_sdf_grid_extent: float = 0.30
  """Half-side of the object-local SDF box, in metres. Box is
  [-extent, extent]^3 (so total side = 2 * extent, default 0.30 m → 0.60 m
  cube). Should comfortably enclose the object plus ~5 cm margin so SDF
  gradients near the surface are well-defined. Voxel size is
  2 * extent / N where N is `obj_sdf_grid_n`.
  """
  obj_sdf_grid_n: int = 48
  """SDF grid resolution per axis (cube of N^3 voxels). Default 48 →
  ~6.25 mm voxels at the default 0.30 m extent — fine enough for sub-cm
  fingertip-surface tracking after trilinear interpolation, cheap on memory
  (~440 KB per side for sdf+grad).
  """
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
  xfrc_omega_rot: float = 0.0
  """If > 0, switches xfrc rotation PD to anisotropic inertia-tensor mode:
      τ = ω² · I_world(t) · axis_angle(ref_q sim_q⁻¹)
        + 2 · ζ_rot · ω · I_world(t) · (ref_angvel - sim_angvel)
  where I_world = R_sim · diag(body_inertia) · R_sim^T (body_inertia is
  MuJoCo's diagonal in the body principal-axis frame; R_sim from sim_obj_quat_w).
  Anisotropic by axis: same response time τ_settling = 1/ω_rot in every axis,
  regardless of object shape. Overrides `xfrc_kp_rot`/`xfrc_kv_rot` scalars.
  """
  xfrc_zeta_rot: float = 1.0
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
  init_noise_scale: float = 1.0
  """Multiplier on the ManipTrans-matched warm-start init noise applied at
  every env reset (wrist trans ±1cm, wrist rot ±10°, finger DOFs ±range/8,
  wrist vel ±1cm/s, finger vel ±0.1). Default 1.0 = original behavior.
  Eval / rollout scripts should set 0.0 — DR perturbation on the ref pose
  reliably tips fingertip-precision contact tasks into visible penetration
  on frame 0. The noise can't be an EventTermCfg (mjlab events fire before
  command_manager.reset, so they'd be overwritten by _resample_command's
  ref-pose write); has to be in-line in the command term."""

  def build(self, env: ManagerBasedRlEnv) -> ManipTransCommand:
    return ManipTransCommand(self, env)
