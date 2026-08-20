"""Non-GUI tests for the OSLFullBodyRobot tuning viewer."""

from pathlib import Path

import mujoco
import numpy as np

from scripts import tune_osl_fullbody_robot as tuning


ROBOT_XML = Path("models/osl_fullbody/body/osl_fullbody_robot.xml").resolve()


def test_qpos0_projection_and_fixed_root_target_are_constraint_consistent():
    full_model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    target, label = tuning.load_target_qpos(full_model, None, 0)
    root_id = mujoco.mj_name2id(full_model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_qadr = int(full_model.jnt_qposadr[root_id])
    original_height = target[root_qadr + 2]
    target = tuning.offset_root_height(full_model, target, 0.075)
    model, fixed_target = tuning.compile_test_model(ROBOT_XML, full_model, target, True)

    assert label == "qpos0"
    assert target[root_qadr + 2] == original_height + 0.075
    assert model.nq == full_model.nq - 7
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root") == -1
    full_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Full Body")
    assert model.body_pos[full_body_id, 2] == target[root_qadr + 2]
    assert tuning.equality_residual(model, fixed_target) < 1e-8


def test_vertical_offset_moves_free_root_target_without_mutating_input():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    target, _ = tuning.load_target_qpos(model, None, 0)
    original = target.copy()
    shifted = tuning.offset_root_height(model, target, -0.025)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_qadr = int(model.jnt_qposadr[root_id])

    np.testing.assert_array_equal(target, original)
    assert shifted[root_qadr + 2] == original[root_qadr + 2] - 0.025
    np.testing.assert_allclose(shifted[: root_qadr + 2], original[: root_qadr + 2])
    np.testing.assert_allclose(shifted[root_qadr + 3 :], original[root_qadr + 3 :])


def test_target_controls_and_in_memory_parameter_overrides():
    full_model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    full_target, _ = tuning.load_target_qpos(full_model, None, 0)
    model, target = tuning.compile_test_model(ROBOT_XML, full_model, full_target, True)
    tuning.apply_parameter_overrides(
        model,
        ["torso=410", "flexion_l=25"],
        ["torso=35"],
        ["arms=150"],
    )
    controls = tuning.position_controls_for_target(model, target)
    data = mujoco.MjData(model)
    data.qpos[:] = target
    data.ctrl[:27] = controls
    mujoco.mj_forward(model, data)

    np.testing.assert_allclose(-model.actuator_biasprm[:3, 1], 410)
    np.testing.assert_allclose(-model.actuator_biasprm[:3, 2], 35)
    np.testing.assert_allclose(model.actuator_forcerange[3:17, 1], 150)
    assert -model.actuator_biasprm[16, 1] == 25
    np.testing.assert_allclose(data.actuator_force[:27], 0, atol=2e-4)


def test_osl_diagnostic_position_controller_clips_in_joint_torque_units():
    full_model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    full_target, _ = tuning.load_target_qpos(full_model, None, 0)
    model, target = tuning.compile_test_model(ROBOT_XML, full_model, full_target, True)
    data = mujoco.MjData(model)
    data.qpos[:] = target
    knee_joint = int(model.actuator_trnid[27, 0])
    knee_qadr = int(model.jnt_qposadr[knee_joint])
    data.qpos[knee_qadr] -= 1.0

    tuning.osl_position_control(model, data, target, 300, 30, 100, 120)

    assert model.actuator_gear[27, 0] * data.ctrl[27] == 100
    assert data.ctrl[28] == 0


def test_ghost_uses_native_render_mesh_ids_and_goal_visual_group():
    full_model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    full_target, _ = tuning.load_target_qpos(full_model, None, 0)
    model, target = tuning.compile_test_model(ROBOT_XML, full_model, full_target, True)
    target_data = mujoco.MjData(model)
    target_data.qpos[:] = target
    mujoco.mj_forward(model, target_data)
    target_scene = tuning.build_target_ghost_scene(model, target_data)

    class ViewerStub:
        user_scn = mujoco.MjvScene(model, maxgeom=1000)

    rgba = np.array([0.471, 0.38, 0.812, 0.5], dtype=np.float32)
    viewer = ViewerStub()
    tuning.add_target_ghost(viewer, model, target_scene, rgba)

    expected = [
        target_scene.geoms[index]
        for index in range(target_scene.ngeom)
        if target_scene.geoms[index].objtype == mujoco.mjtObj.mjOBJ_GEOM
        and target_scene.geoms[index].objid >= 0
        and model.geom_group[target_scene.geoms[index].objid] == 0
    ]
    assert viewer.user_scn.ngeom == len(expected)
    for index, source in enumerate(expected):
        ghost = viewer.user_scn.geoms[index]
        assert ghost.objid == source.objid
        assert ghost.dataid == source.dataid
        np.testing.assert_allclose(ghost.pos, source.pos)
        np.testing.assert_allclose(ghost.mat, source.mat)
        np.testing.assert_allclose(ghost.rgba, rgba)
