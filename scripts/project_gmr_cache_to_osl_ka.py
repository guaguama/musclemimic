"""Project MyoFullBody GMR-cached trajectories onto MyoLeg80_OSL_KA.

For every MyoFullBody/gmr/<motion>.npz under the user's cache root, build an
OSL_KA-compatible cache by name-mapping qpos/qvel into OSL_KA's joint vector
and writing a minimal .npz to the OSL_KA cache directory. The trajectory
loader's extend_motion() fills in FK-derived fields (xpos, site_xpos, ...)
from qpos using OSL_KA's MjModel at first env load.

Default behavior: project only the motions in
KIT_KINESIS_TRAINING_MOTIONS_MINT_STRAIGHT_FORWARDS.

Examples:
  uv run scripts/project_gmr_cache_to_osl_ka.py
  uv run scripts/project_gmr_cache_to_osl_ka.py --motion KIT/314/walking_medium09_poses
  uv run scripts/project_gmr_cache_to_osl_ka.py --all
  uv run scripts/project_gmr_cache_to_osl_ka.py --force
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np

import musclemimic.environments  # noqa: F401  triggers env registration
from loco_mujoco.smpl.const import KIT_KINESIS_TRAINING_MOTIONS_MINT_STRAIGHT_FORWARDS
from musclemimic.environments import LocoEnv
from musclemimic.utils.gmr_cache import resolve_gmr_cache_root

SOURCE_ENV = "MyoFullBody"
TARGET_ENV = "MyoLeg80_OSL_KA"

JOINT_RENAME = {
    "hip_flexion_r": "hip_flexion_r",
    "hip_adduction_r": "hip_adduction_r",
    "hip_rotation_r": "hip_rotation_r",
    "knee_angle_r": "osl_knee_angle_r",
    "ankle_angle_r": "osl_ankle_angle_r",
    "hip_flexion_l": "hip_flexion_l",
    "hip_adduction_l": "hip_adduction_l",
    "hip_rotation_l": "hip_rotation_l",
    "knee_angle_translation2_l": "knee_angle_l_translation2",
    "knee_angle_translation1_l": "knee_angle_l_translation1",
    "knee_angle_l": "knee_angle_l",
    "knee_angle_rotation2_l": "knee_angle_l_rotation2",
    "knee_angle_rotation3_l": "knee_angle_l_rotation3",
    "ankle_angle_l": "ankle_angle_l",
    "subtalar_angle_l": "subtalar_angle_l",
    "mtp_angle_l": "mtp_angle_l",
    "knee_angle_beta_translation2_l": "knee_angle_l_beta_translation2",
    "knee_angle_beta_translation1_l": "knee_angle_l_beta_translation1",
    "knee_angle_beta_rotation1_l": "knee_angle_l_beta_rotation1",
}

# Sign flips (verify at first preview, fill in here).
QPOS_SIGN: dict[str, float] = {
    # "osl_knee_angle_r": -1.0,
    # "osl_ankle_angle_r": -1.0,
}

# Pelvis-frame correction. MyoFullBody chain's pelvis is euler="1.57 -1.57 0";
# OSL_KA chain's pelvis is quat="0.707107 0.707107 0 0". The same source freejoint
# qpos[3:7] therefore renders the OSL_KA pelvis at a different world orientation.
# q_correction = q_MFB_pelvis * q_OSL_pelvis^-1 = -90° about Z. Right-multiply
# the source quat by this constant. [w, x, y, z] order.
PELVIS_QUAT_CORRECTION = np.array([0.7071068, 0.0, 0.0, -0.7071068], dtype=np.float64)


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for batched quats in MuJoCo [w, x, y, z] order.

    q1 shape (T, 4), q2 shape (4,) or (T, 4). Returns (T, 4).
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


_QSIZE = {0: 7, 1: 4, 2: 1, 3: 1}  # free, ball, slide, hinge
_VSIZE = {0: 6, 1: 3, 2: 1, 3: 1}


def _build_layout_from_model(model: mujoco.MjModel):
    qpos_off, qvel_off = {}, {}
    qp = qv = 0
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jt = int(model.jnt_type[j])
        qpos_off[name] = (qp, qp + _QSIZE[jt])
        qvel_off[name] = (qv, qv + _VSIZE[jt])
        qp += _QSIZE[jt]
        qv += _VSIZE[jt]
    return qpos_off, qvel_off


def _build_layout_from_arrays(joint_names, jnt_types):
    qpos_off, qvel_off = {}, {}
    qp = qv = 0
    for n, t in zip(joint_names, jnt_types):
        t = int(t)
        qpos_off[n] = (qp, qp + _QSIZE[t])
        qvel_off[n] = (qv, qv + _VSIZE[t])
        qp += _QSIZE[t]
        qv += _VSIZE[t]
    return qpos_off, qvel_off


def _compute_fk_fields(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    mimic_site_ids: np.ndarray,
):
    """Run mj_forward per frame; collect body FK + site FK filtered to mimic sites."""
    T = qpos.shape[0]
    n_mimic = len(mimic_site_ids)
    data = mujoco.MjData(model)
    xpos = np.empty((T, model.nbody, 3), dtype=np.float32)
    xquat = np.empty((T, model.nbody, 4), dtype=np.float32)
    cvel = np.empty((T, model.nbody, 6), dtype=np.float32)
    subtree_com = np.empty((T, model.nbody, 3), dtype=np.float32)
    site_xpos = np.empty((T, n_mimic, 3), dtype=np.float32)
    site_xmat = np.empty((T, n_mimic, 9), dtype=np.float32)
    for t in range(T):
        data.qpos[:] = qpos[t]
        data.qvel[:] = qvel[t]
        mujoco.mj_forward(model, data)
        xpos[t] = data.xpos
        xquat[t] = data.xquat
        cvel[t] = data.cvel
        subtree_com[t] = data.subtree_com
        site_xpos[t] = data.site_xpos[mimic_site_ids]
        site_xmat[t] = data.site_xmat[mimic_site_ids]
    return xpos, xquat, cvel, subtree_com, site_xpos, site_xmat


def project_one(src_npz: Path, tgt_npz: Path, src_layout, tgt_layout, tgt_model, mimic_site_names):
    src = np.load(src_npz, allow_pickle=True)
    src_qpos = src["qpos"].astype(np.float32)
    src_qvel = src["qvel"].astype(np.float32)
    T = src_qpos.shape[0]

    tgt_qpos = np.zeros((T, tgt_model.nq), dtype=np.float32)
    tgt_qvel = np.zeros((T, tgt_model.nv), dtype=np.float32)

    # Position copies directly. Quaternion is right-multiplied by the pelvis-frame
    # correction so the OSL_KA pelvis renders at the same world orientation as
    # MyoFullBody's. qvel is unchanged: linear velocity is in world frame and
    # angular velocity is in the body's local frame, both invariant under the
    # constant body-frame rebase.
    tgt_qpos[:, 0:3] = src_qpos[:, 0:3]
    tgt_qpos[:, 3:7] = _quat_mul_wxyz(
        src_qpos[:, 3:7].astype(np.float64), PELVIS_QUAT_CORRECTION
    ).astype(np.float32)
    tgt_qvel[:, 0:6] = src_qvel[:, 0:6]

    src_qpos_off, src_qvel_off = src_layout
    tgt_qpos_off, tgt_qvel_off = tgt_layout

    for src_name, tgt_name in JOINT_RENAME.items():
        if src_name not in src_qpos_off or tgt_name not in tgt_qpos_off:
            continue  # joint absent in this source/target; skip silently
        sa, sb = src_qpos_off[src_name]
        ta, tb = tgt_qpos_off[tgt_name]
        sva, svb = src_qvel_off[src_name]
        tva, tvb = tgt_qvel_off[tgt_name]
        sign = QPOS_SIGN.get(tgt_name, 1.0)
        tgt_qpos[:, ta:tb] = src_qpos[:, sa:sb] * sign
        tgt_qvel[:, tva:tvb] = src_qvel[:, sva:svb] * sign

    if "split_points" in src.files:
        split_points = src["split_points"].astype(np.int64)
    else:
        split_points = np.array([0, T], dtype=np.int64)

    frequency = float(src["frequency"]) if "frequency" in src.files else 30.0

    md = src["metadata"].item() if "metadata" in src.files else {}
    md = dict(md) if isinstance(md, dict) else {}
    md["projected_from"] = SOURCE_ENV
    md["projection_script"] = "project_gmr_cache_to_osl_ka.py"

    mimic_site_ids = np.array(
        [mujoco.mj_name2id(tgt_model, mujoco.mjtObj.mjOBJ_SITE, n) for n in mimic_site_names],
        dtype=np.int32,
    )
    if (mimic_site_ids < 0).any():
        missing = [n for n, i in zip(mimic_site_names, mimic_site_ids) if i < 0]
        raise RuntimeError(f"Mimic sites not found in model: {missing}")

    static = _extract_static_model_metadata(tgt_model, mimic_site_names, mimic_site_ids)
    xpos, xquat, cvel, subtree_com, site_xpos, site_xmat = _compute_fk_fields(
        tgt_model, tgt_qpos, tgt_qvel, mimic_site_ids
    )

    tgt_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        tgt_npz,
        qpos=tgt_qpos,
        qvel=tgt_qvel,
        frequency=np.float32(frequency),
        split_points=split_points,
        metadata=md,
        xpos=xpos,
        xquat=xquat,
        cvel=cvel,
        subtree_com=subtree_com,
        site_xpos=site_xpos,
        site_xmat=site_xmat,
        **static,
    )


def _extract_static_model_metadata(
    model: mujoco.MjModel,
    mimic_site_names: list[str],
    mimic_site_ids: np.ndarray,
) -> dict:
    """Joint metadata + full body table + filtered mimic-site metadata.

    Convention matches existing MyoFullBody/gmr cache files: body_* contains
    the full body table; site_* is filtered to only the mimic sites.
    """
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]
    return dict(
        njnt=np.int32(model.njnt),
        jnt_type=np.array(model.jnt_type[: model.njnt], dtype=np.int32),
        joint_names=np.array(joint_names),
        nbody=np.int32(model.nbody),
        body_names=np.array(body_names),
        body_rootid=np.array(model.body_rootid[: model.nbody], dtype=np.int32),
        body_weldid=np.array(model.body_weldid[: model.nbody], dtype=np.int32),
        body_mocapid=np.array(model.body_mocapid[: model.nbody], dtype=np.int32),
        body_pos=np.array(model.body_pos[: model.nbody], dtype=np.float32),
        body_quat=np.array(model.body_quat[: model.nbody], dtype=np.float32),
        body_ipos=np.array(model.body_ipos[: model.nbody], dtype=np.float32),
        body_iquat=np.array(model.body_iquat[: model.nbody], dtype=np.float32),
        nsite=np.int32(len(mimic_site_names)),
        site_names=np.array(mimic_site_names),
        site_bodyid=np.array(model.site_bodyid[mimic_site_ids], dtype=np.int32),
        site_pos=np.array(model.site_pos[mimic_site_ids], dtype=np.float32),
        site_quat=np.array(model.site_quat[mimic_site_ids], dtype=np.float32),
    )


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--motion", action="append", default=None,
                    help="Specific motion path (repeatable).")
    ap.add_argument("--all", action="store_true",
                    help="Project every motion present in MyoFullBody/gmr/.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing target files.")
    args = ap.parse_args()

    cache_root = Path(resolve_gmr_cache_root())
    src_root = cache_root / SOURCE_ENV / "gmr"
    tgt_root = cache_root / TARGET_ENV / "gmr"
    if not src_root.exists():
        raise SystemExit(f"Source cache not found: {src_root}")

    if args.all:
        motion_paths = sorted(
            p.relative_to(src_root).with_suffix("").as_posix()
            for p in src_root.rglob("*.npz")
        )
    elif args.motion:
        motion_paths = list(args.motion)
    else:
        motion_paths = list(KIT_KINESIS_TRAINING_MOTIONS_MINT_STRAIGHT_FORWARDS)

    print(f"Source: {src_root}")
    print(f"Target: {tgt_root}")
    print(f"Motions: {len(motion_paths)}")

    tgt_env = LocoEnv.registered_envs[TARGET_ENV](headless=True, horizon=10)
    tgt_model = tgt_env.model
    tgt_layout = _build_layout_from_model(tgt_model)
    mimic_site_names = list(tgt_env.body2sites_for_mimic.values())

    sample = next(src_root.rglob("*.npz"))
    sd = np.load(sample, allow_pickle=True)
    src_layout = _build_layout_from_arrays(sd["joint_names"].tolist(), sd["jnt_type"].tolist())

    n_done = n_skipped = n_missing = 0
    for m in motion_paths:
        src_path = src_root / f"{m}.npz"
        tgt_path = tgt_root / f"{m}.npz"
        if not src_path.exists():
            print(f"  [missing] {m}")
            n_missing += 1
            continue
        if tgt_path.exists() and not args.force:
            n_skipped += 1
            continue
        project_one(src_path, tgt_path, src_layout, tgt_layout, tgt_model, mimic_site_names)
        print(f"  [ok]      {m}")
        n_done += 1

    print(f"\nDone: {n_done} written, {n_skipped} skipped (use --force to overwrite), {n_missing} missing.")


if __name__ == "__main__":
    main()
