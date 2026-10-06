"""Tests for the vendored official full-body + OSL model (models/new_osl_fullbody).

Part 1 asserts the *model* only -- no environment class exists yet.  The load
bearing assertion is the rigid/compliant coupling with ``mjDSBL_EULERDAMP``:
the whole reason two variants are generated is that socket compliance is what
forces Euler damping to stay implicit, and the official rigid socket does not.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import mujoco
import numpy as np
import pytest

from scripts import vendor_new_osl_fullbody_model as vendor


TIMESTEP = 0.002  # OSLFullBody.__init__ default; every stability number doubles from the XML's 1 ms
EULERDAMP_BOUND = 2.0  # explicit-Euler stability limit on dof_damping * dt / M_ii


def _model(path: Path) -> mujoco.MjModel:
    # Absolute paths only: MuJoCo 3.4 resolves nested include/texture paths
    # against the main file's directory, so a relative path loses the skybox.
    return mujoco.MjModel.from_xml_path(str(path.resolve()))


def _names(model: mujoco.MjModel, objtype: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, objtype, index) for index in range(count)]


def _worst_damping_ratio(model: mujoco.MjModel, key: int = 0) -> float:
    """max(dof_damping * dt / M_ii) -- diverges under explicit Euler above 2."""
    model.opt.timestep = TIMESTEP
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    inertia = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, inertia, data.qM)
    return float((model.dof_damping * TIMESTEP / np.diag(inertia)).max())


@contextlib.contextmanager
def _tolerating_mujoco_warnings():
    """Suspend loco_mujoco's global warning-to-exception hook.

    ``Mujoco.__init__`` installs ``user_warning_raise_exception`` process-wide
    (loco_mujoco/core/mujoco_base.py:187) and never removes it, so once any other
    test has built an environment, a deliberately unstable rollout here would
    raise instead of being counted.
    """
    previous = mujoco.get_mju_user_warning()
    # A no-op handler rather than None: with no handler at all MuJoCo falls back
    # to printing the warning and writing MUJOCO_LOG.TXT into the repo root.
    mujoco.set_mju_user_warning(lambda _warning: None)
    try:
        yield
    finally:
        mujoco.set_mju_user_warning(previous)


def _bad_qacc(model: mujoco.MjModel, *, disable_eulerdamp: bool, steps: int = 3000) -> int:
    """Instability counter after a free fall from the ``stand`` keyframe.

    Uses mjWARN_BADQACC rather than isfinite(qvel): MuJoCo clamps qacc when the
    integrator blows up, so a diverging model keeps finite velocities and a
    finiteness check silently passes.
    """
    flags = int(model.opt.disableflags)
    model.opt.timestep = TIMESTEP
    if disable_eulerdamp:
        model.opt.disableflags = flags | int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    try:
        with _tolerating_mujoco_warnings():
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, 0)
            for _ in range(steps):
                mujoco.mj_step(model, data)
        return int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
    finally:
        model.opt.disableflags = flags


@pytest.fixture(scope="module")
def rigid_model() -> mujoco.MjModel:
    return _model(vendor.RIGID_XML)


@pytest.fixture(scope="module")
def compliant_model() -> mujoco.MjModel:
    return _model(vendor.COMPLIANT_XML)


@pytest.mark.skipif(
    not vendor.DEFAULT_SOURCE_XML.is_file(),
    reason="the assist_sim export is a local artifact of the myoassist checkout",
)
def test_vendored_tree_is_a_deterministic_regeneration_of_the_export():
    subprocess.run(
        [sys.executable, str(Path(vendor.__file__).resolve()), "--check"],
        cwd=vendor.REPO_ROOT,
        check=True,
    )


@pytest.mark.parametrize("path", [vendor.RIGID_XML, vendor.COMPLIANT_XML])
def test_assets_are_local_and_complete(path):
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    references = [element.get("file") for element in root.iter("mesh") if element.get("file")]
    assert references, "no asset references survived vendoring"
    assert not any(os.path.isabs(reference) for reference in references)
    assert not any(reference.startswith("..") for reference in references)

    # The scene is the one deliberate escape, and it stays inside the model tree.
    includes = [element.get("file") for element in root.iter("include")]
    assert includes == ["../scene/musclemimic_scene.xml"]

    meshes = vendor.OUTPUT_ROOT / "meshes"
    missing = sorted({name for name in references if not (meshes / name).is_file()})
    assert not missing, f"missing vendored meshes: {missing}"
    # No orphans: every tracked STL is referenced by the model.
    assert {path.name for path in meshes.glob("*")} == set(references)


def test_rigid_variant_matches_the_official_export(rigid_model):
    dimensions = (
        rigid_model.nq,
        rigid_model.nv,
        rigid_model.nu,
        rigid_model.nbody,
        rigid_model.neq,
        rigid_model.na,
        rigid_model.nkey,
    )
    assert dimensions == vendor.RIGID_DIMENSIONS
    assert rigid_model.body_mass.sum() == pytest.approx(vendor.EXPECTED_MASS, abs=1e-4)
    assert rigid_model.npair == 46
    assert _names(rigid_model, mujoco.mjtObj.mjOBJ_KEY, rigid_model.nkey) == [
        "stand",
        "walk_left",
        "walk_right",
        "squat",
        "lunge",
    ]
    # The socket is welded upstream, and nothing else in the model is sprung.
    assert not rigid_model.jnt_stiffness.any()
    actuators = _names(rigid_model, mujoco.mjtObj.mjOBJ_ACTUATOR, rigid_model.nu)
    motors = [
        name
        for name, dyntype in zip(actuators, rigid_model.actuator_dyntype)
        if dyntype != mujoco.mjtDyn.mjDYN_MUSCLE
    ]
    assert motors == ["osl_knee_torque_actuator", "osl_ankle_torque_actuator"]


def test_scene_supplies_floor_light_and_file_backed_skybox(rigid_model):
    geoms = _names(rigid_model, mujoco.mjtObj.mjOBJ_GEOM, rigid_model.ngeom)
    assert geoms.count("floor") == 1
    floor = geoms.index("floor")
    assert rigid_model.geom_type[floor] == mujoco.mjtGeom.mjGEOM_PLANE
    # The feet are contype=1/conaffinity=0, so the floor must answer with conaffinity=1.
    assert rigid_model.geom_conaffinity[floor] == 1
    assert rigid_model.nlight >= 1

    textures = _names(rigid_model, mujoco.mjtObj.mjOBJ_TEXTURE, rigid_model.ntex)
    assert textures.count("sky") == 1  # the export's builtin gradient must be gone
    sky = textures.index("sky")
    assert rigid_model.tex_pathadr[sky] >= 0, "skybox is not file-backed"


def test_compliant_variant_only_adds_the_socket_dofs(rigid_model, compliant_model):
    dimensions = (compliant_model.nq, compliant_model.nv, compliant_model.na)
    assert dimensions == vendor.COMPLIANT_DIMENSIONS[:2] + vendor.COMPLIANT_DIMENSIONS[5:6]
    assert compliant_model.body_mass.sum() == pytest.approx(
        rigid_model.body_mass.sum(), abs=1e-9
    ), "socket joints must not change mass"

    for objtype, count in (
        (mujoco.mjtObj.mjOBJ_BODY, "nbody"),
        (mujoco.mjtObj.mjOBJ_GEOM, "ngeom"),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, "nu"),
        (mujoco.mjtObj.mjOBJ_SENSOR, "nsensor"),
        (mujoco.mjtObj.mjOBJ_SITE, "nsite"),
    ):
        rigid_names = _names(rigid_model, objtype, getattr(rigid_model, count))
        compliant_names = _names(compliant_model, objtype, getattr(compliant_model, count))
        assert rigid_names == compliant_names, f"{count} differs between variants"

    rigid_joints = _names(rigid_model, mujoco.mjtObj.mjOBJ_JOINT, rigid_model.njnt)
    compliant_joints = _names(compliant_model, mujoco.mjtObj.mjOBJ_JOINT, compliant_model.njnt)
    assert set(compliant_joints) - set(rigid_joints) == set(vendor.SOCKET_JOINT_NAMES)

    # The four DoFs land immediately before the prosthetic knee, which is what
    # makes the spliced keyframes line up with the joints they describe.
    knee = mujoco.mj_name2id(rigid_model, mujoco.mjtObj.mjOBJ_JOINT, vendor.JOINT_AFTER_SOCKET)
    base = int(rigid_model.jnt_qposadr[knee])
    for offset, name in enumerate(vendor.SOCKET_JOINT_NAMES):
        joint = mujoco.mj_name2id(compliant_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert int(compliant_model.jnt_qposadr[joint]) == base + offset
    shifted = mujoco.mj_name2id(
        compliant_model, mujoco.mjtObj.mjOBJ_JOINT, vendor.JOINT_AFTER_SOCKET
    )
    assert int(compliant_model.jnt_qposadr[shifted]) == base + len(vendor.SOCKET_JOINT_NAMES)


def test_socket_compliance_parameters_match_the_hand_built_model(compliant_model):
    expected = {
        "socket_piston": (mujoco.mjtJoint.mjJNT_SLIDE, 20000.0, 10000.0, 0.035),
        "socket_rotation_1": (mujoco.mjtJoint.mjJNT_HINGE, 350.0, 175.0, 0.122173),
        "socket_rotation_2": (mujoco.mjtJoint.mjJNT_HINGE, 350.0, 175.0, 0.122173),
        "socket_rotation_3": (mujoco.mjtJoint.mjJNT_HINGE, 350.0, 175.0, 0.122173),
    }
    for name, (jnt_type, stiffness, damping, limit) in expected.items():
        joint = mujoco.mj_name2id(compliant_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint >= 0, name
        assert compliant_model.jnt_type[joint] == jnt_type
        assert compliant_model.jnt_stiffness[joint] == pytest.approx(stiffness)
        assert compliant_model.jnt_range[joint] == pytest.approx([-limit, limit])
        dof = int(compliant_model.jnt_dofadr[joint])
        assert compliant_model.dof_damping[dof] == pytest.approx(damping)
        assert compliant_model.dof_armature[dof] == pytest.approx(0.01)


def test_euler_damping_is_needed_only_for_the_compliant_socket(rigid_model, compliant_model):
    """The claim Part 2 relies on to drop the OSL environments' EULERDAMP workaround."""
    rigid_ratio = _worst_damping_ratio(rigid_model)
    compliant_ratio = _worst_damping_ratio(compliant_model)
    assert rigid_ratio < 1.0, f"rigid socket no longer inside the Euler bound: {rigid_ratio}"
    assert compliant_ratio > EULERDAMP_BOUND, compliant_ratio

    assert _bad_qacc(rigid_model, disable_eulerdamp=True) == 0
    assert _bad_qacc(compliant_model, disable_eulerdamp=True) > 0
    assert _bad_qacc(compliant_model, disable_eulerdamp=False) == 0


def test_model_settles_onto_the_floor(rigid_model):
    rigid_model.opt.timestep = TIMESTEP
    data = mujoco.MjData(rigid_model)
    mujoco.mj_resetDataKeyframe(rigid_model, data, 0)
    for _ in range(2000):
        mujoco.mj_step(rigid_model, data)
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
    assert data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 0
    assert 0.0 < data.qpos[2] < 1.0, "root never fell onto the floor"
    assert data.ncon > 0


@pytest.mark.integration
def test_warp_backend_stability_without_euler_damping():
    """CPU MuJoCo is not proof: the original NaN divergence was seen under Warp.

    Single-world on purpose.  Divergence is a per-world numerical property, and
    hand-batching an ``impl="warp"`` Data is not a supported operation -- the real
    batching happens inside ``MjxMujoco`` (musclemimic/core/mujoco_mjx.py:87-118),
    which does not exist for this model until Part 2.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from mujoco import mjx

    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("the warp backend requires a GPU")

    def roll(path, *, disable_eulerdamp, steps=2000):
        model = _model(path)
        model.opt.timestep = TIMESTEP
        if disable_eulerdamp:
            model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)

        # make_data, not put_data: MJX-Warp cannot build Data from an MjData.
        # Budgets mirror MyoFullBody._modify_spec_for_mjx (myofullbody.py:420-431).
        mjx_model = mjx.put_model(model, impl="warp")
        mjx_data = mjx.make_data(model, impl="warp", nconmax=96, njmax=768, naconmax=96)
        mjx_data = mjx_data.replace(
            qpos=jnp.array(data.qpos), qvel=jnp.array(data.qvel), act=jnp.array(data.act)
        )
        step = jax.jit(mjx.step)

        key = jax.random.PRNGKey(0)
        peak_contacts = 0
        for index in range(steps):
            if index % 50 == 0:
                key, subkey = jax.random.split(key)
                mjx_data = mjx_data.replace(ctrl=jax.random.uniform(subkey, (model.nu,)))
            mjx_data = step(mjx_model, mjx_data)
            if index % 100 == 0:
                peak_contacts = max(
                    peak_contacts, int(jnp.asarray(mjx_data._impl.nacon).reshape(-1)[0])
                )
                if not bool(
                    jnp.all(jnp.isfinite(mjx_data.qvel)) & jnp.all(jnp.isfinite(mjx_data.qpos))
                ):
                    return False, peak_contacts
        return True, peak_contacts

    rigid_finite, rigid_contacts = roll(vendor.RIGID_XML, disable_eulerdamp=True)
    assert rigid_finite, "rigid socket diverged under warp without Euler damping"
    # The claim's converse, so a future re-export cannot pass this test vacuously.
    assert not roll(vendor.COMPLIANT_XML, disable_eulerdamp=True)[0]
    assert roll(vendor.COMPLIANT_XML, disable_eulerdamp=False)[0]

    # myofullbody.py:422 budgets 96 contacts per env; exceeding it silently drops contacts.
    assert rigid_contacts <= 96, rigid_contacts
