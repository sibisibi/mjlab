"""Object entity from COACD convex decomposition meshes.

Creates an mjlab EntityCfg for a free-floating rigid object using:
- visual.obj for rendering (no collision)
- convex/*.obj for collision (contype=2, conaffinity=3)

`mesh_scale` is a generic per-call parameter. Dataset-specific corrections
(e.g. OakInk2's per-object scale table) live in the dataset preprocessing
scripts under src/preprocess/dataset/ and are propagated to training via
the `*_object_mesh_scale` fields in `task_info.json`.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


def get_object_spec(
  obj_dir: str | Path,
  name: str,
  density: float,
  mesh_scale: float = 1.0,
) -> mujoco.MjSpec:
  """Build MjSpec for an object with convex collision meshes.

  `mesh_scale` scales all visual + convex meshes uniformly at compile time.
  Mass scales as scale^3 (density unchanged). Mass cap at 0.5 kg still applies.
  """
  obj_dir = Path(obj_dir)
  spec = mujoco.MjSpec()
  spec.modelname = name

  body = spec.worldbody.add_body(name=name)
  body.add_freejoint(name=f"{name}_freejoint")

  # Visual mesh
  visual_path = obj_dir / "visual.obj"
  visual_mesh_name = f"{name}_visual"
  mesh = spec.add_mesh()
  mesh.name = visual_mesh_name
  mesh.file = f"{visual_mesh_name}.obj"
  mesh.scale = (mesh_scale, mesh_scale, mesh_scale)
  spec.assets[f"{visual_mesh_name}.obj"] = visual_path.read_bytes()
  body.add_geom(
    name=f"{name}_visual_geom",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname=visual_mesh_name,
    contype=0,
    conaffinity=0,
    rgba=(0.8, 0.6, 0.4, 1.0),
    group=0,
    mass=0,
  )

  # Convex collision meshes
  convex_dir = obj_dir / "convex"
  convex_files = sorted(convex_dir.glob("*.obj"), key=lambda p: int(p.stem))
  for obj_file in convex_files:
    mesh_name = f"{name}_convex_{obj_file.stem}"
    mesh = spec.add_mesh()
    mesh.name = mesh_name
    mesh.file = f"{mesh_name}.obj"
    mesh.scale = (mesh_scale, mesh_scale, mesh_scale)
    spec.assets[f"{mesh_name}.obj"] = obj_file.read_bytes()
    body.add_geom(
      name=f"{name}_col_{obj_file.stem}",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=mesh_name,
      contype=2,
      conaffinity=3,
      condim=3,
      friction=(2.0, 0.05, 0.05),
      rgba=(0.0, 0.0, 0.0, 0.0),
      group=3,
      density=density,
    )

  # Cap total mass at 0.5 kg (ManipTrans convention)
  m = spec.compile()
  total_mass = m.body_mass[1]  # body 0 is world, body 1 is the object
  if total_mass > 0.5:
    density_scale = 0.5 / total_mass
    for geom in spec.geoms:
      if geom.contype == 2:
        geom.density = density * density_scale

  return spec


def get_object_cfg(
  obj_dir: str | Path,
  name: str,
  density: float,
  mesh_scale: float = 1.0,
) -> EntityCfg:
  """Create EntityCfg for a free-floating object.

  `mesh_scale` scales the object geometry (see `get_object_spec`).
  """
  obj_dir_str = str(Path(obj_dir))
  return EntityCfg(
    spec_fn=lambda d=obj_dir_str, n=name, dn=density, ms=mesh_scale: get_object_spec(d, n, dn, ms),
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.5),
      rot=(1.0, 0.0, 0.0, 0.0),
    ),
  )


# Joint order must stay consistent with _update_command's ctrl write order.
# (3 slides, then 3 hinges, axes [x, y, z].)
_ACTUATED_JOINT_SUFFIXES = ("pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z")


def get_actuated_object_spec(
  obj_dir: str | Path,
  name: str,
  density: float,
  kp_pos: float,
  kv_pos: float,
  kp_rot: float,
  kv_rot: float,
  joint_armature_pos: float = 0.0,
  joint_armature_rot: float = 0.0,
  joint_damping: float = 0.0,
  mesh_scale: float = 1.0,
) -> mujoco.MjSpec:
  """Variant of `get_object_spec` where the freejoint is replaced by 6 individual
  slide/hinge joints with position actuators. This is the Spider `mjwp_act` path:
  the object is held to a per-step reference pose by MuJoCo position actuators
  (native stable pipeline), with configurable kp/kv that can be decayed over
  training via a curriculum.
  """
  obj_dir = Path(obj_dir)
  spec = mujoco.MjSpec()
  spec.modelname = name

  body = spec.worldbody.add_body(name=name)

  joint_defs = (
    ("pos_x", mujoco.mjtJoint.mjJNT_SLIDE, (1, 0, 0), "pos"),
    ("pos_y", mujoco.mjtJoint.mjJNT_SLIDE, (0, 1, 0), "pos"),
    ("pos_z", mujoco.mjtJoint.mjJNT_SLIDE, (0, 0, 1), "pos"),
    ("rot_x", mujoco.mjtJoint.mjJNT_HINGE, (1, 0, 0), "rot"),
    ("rot_y", mujoco.mjtJoint.mjJNT_HINGE, (0, 1, 0), "rot"),
    ("rot_z", mujoco.mjtJoint.mjJNT_HINGE, (0, 0, 1), "rot"),
  )
  for jname, jtype, axis, kind in joint_defs:
    arm = joint_armature_pos if kind == "pos" else joint_armature_rot
    body.add_joint(
      name=f"{name}_{jname}",
      type=jtype,
      axis=axis,
      armature=arm,
      damping=joint_damping,
    )

  # Visual mesh
  visual_path = obj_dir / "visual.obj"
  v_mesh = spec.add_mesh()
  v_mesh.name = f"{name}_visual"
  v_mesh.file = f"{name}_visual.obj"
  v_mesh.scale = (mesh_scale, mesh_scale, mesh_scale)
  spec.assets[f"{name}_visual.obj"] = visual_path.read_bytes()
  body.add_geom(
    name=f"{name}_visual_geom",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname=f"{name}_visual",
    contype=0,
    conaffinity=0,
    rgba=(0.8, 0.6, 0.4, 1.0),
    group=0,
    mass=0,
  )

  # Convex collision meshes — same contype/conaffinity/friction as freejoint path
  # so the env-level collision matrix is unchanged.
  convex_dir = obj_dir / "convex"
  convex_files = sorted(convex_dir.glob("*.obj"), key=lambda p: int(p.stem))
  for obj_file in convex_files:
    mesh_name = f"{name}_convex_{obj_file.stem}"
    mesh = spec.add_mesh()
    mesh.name = mesh_name
    mesh.file = f"{mesh_name}.obj"
    mesh.scale = (mesh_scale, mesh_scale, mesh_scale)
    spec.assets[f"{mesh_name}.obj"] = obj_file.read_bytes()
    body.add_geom(
      name=f"{name}_col_{obj_file.stem}",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=mesh_name,
      contype=2,
      conaffinity=3,
      condim=3,
      friction=(2.0, 0.05, 0.05),
      rgba=(0.0, 0.0, 0.0, 0.0),
      group=3,
      density=density,
    )

  # Cap total mass at 0.5 kg (ManipTrans convention, same as freejoint variant).
  m = spec.compile()
  total_mass = m.body_mass[1]
  if total_mass > 0.5:
    density_scale = 0.5 / total_mass
    for geom in spec.geoms:
      if geom.contype == 2:
        geom.density = density * density_scale

  # Actuators — position controllers on each of the 6 joints.
  for jname, _, _, kind in joint_defs:
    act = spec.add_actuator()
    act.name = f"{name}_{jname}"
    act.target = f"{name}_{jname}"
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    kp = kp_pos if kind == "pos" else kp_rot
    kv = kv_pos if kind == "pos" else kv_rot
    act.set_to_position(kp=kp, kv=kv)

  return spec


def get_actuated_object_cfg(
  obj_dir: str | Path,
  name: str,
  density: float,
  kp_pos: float,
  kv_pos: float,
  kp_rot: float,
  kv_rot: float,
  mesh_scale: float = 1.0,
) -> EntityCfg:
  """Create EntityCfg for an object held by 6 position actuators (Spider `mjwp_act`
  path). Used for single-stage training where the hard freejoint pin is replaced
  by a virtual controller whose gains can be decayed over training.

  `mesh_scale` scales the object geometry (see `get_actuated_object_spec`).
  """
  obj_dir_str = str(Path(obj_dir))
  return EntityCfg(
    spec_fn=lambda d=obj_dir_str, n=name, dn=density, kpp=kp_pos, kvp=kv_pos, kpr=kp_rot, kvr=kv_rot, ms=mesh_scale: get_actuated_object_spec(d, n, dn, kpp, kvp, kpr, kvr, mesh_scale=ms),
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.0),  # No body attach offset — joint_pos writes are
                            # the world-frame pose directly.
      rot=(1.0, 0.0, 0.0, 0.0),
      # Default joint_pos={".*": 0.0} → 6-zero keyframe. _resample_command
      # overwrites qpos and ctrl on reset with the per-env reference pose.
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        XmlActuatorCfg(
          target_names_expr=(f"{name}_pos_x", f"{name}_pos_y", f"{name}_pos_z",
                             f"{name}_rot_x", f"{name}_rot_y", f"{name}_rot_z"),
          transmission_type=TransmissionType.JOINT,
        ),
      ),
    ),
  )
