# osl_fullbody (imported, v3)

Self-contained copy of the official NeuMoVe Lab full-body + OSL knee-ankle
prosthesis model, exported with myo_sim 0.2.3 and with the prosthetic shank
lengthened by hand to match the intact leg. It replaces two earlier models that
now live on the `old_osl` branch: the hand-assembled `osl_fullbody` (v1), which
grafted the `myoLeg80_OSL_KA` leg onto the packaged MyoFullBody model, and the
myo_sim 0.2.2 export `new_osl_fullbody` (v2).

Everything here except `README.md` and `scene/` is generated. Run
`scripts/vendor_osl_fullbody_model.py` for the two muscle models and
`scripts/generate_osl_fullbody_robot.py` for the two robot models rather than
editing the XMLs, and pass `--check` to prove the tracked files match a
deterministic regeneration.

## Provenance

- Upstream repo: [myoassist](https://github.com/neumovelab/myoassist) at commit
  `acee39b7`, fork `github.com/guaguama/myoassist`.
- The export comes from the `assist_sim` 1.0.0 CLI with myo_sim 0.2.3
  (`93b0ca8`), generated 2026-10-14:

  ```bash
  cd <myoassist repo>
  python -m assist_sim combine myofullbody OpenSourceLeg_KA_L1 \
      -o models/osl_fullbody_v3.xml
  ```

  The amputation recipe is `assist_sim/models/OpenSourceLeg/KA_L1config.yaml`
  (body removals, mesh replacements, residual-femur mass override, muscle
  removals and tendon re-anchorings, the two OSL motors, keyframes, sensors).
- Against the myo_sim 0.2.2 export behind v2, the export differs only by the
  0.2.3 fixes. The left-knee polycoefs `knee_angle_translation2_constraint_l`
  and `knee_angle_rotation3_constraint_l` are negated, which corrects a left
  shank that sat up to 61 mm off the mirror of the right at 90 degrees of knee
  flexion. Wrap geoms `DELT1hh_ellipsoid_DELT1` and `EDCL/EDCM_torus_wrap` (both
  arms) moved and `back_cylinder_l` shrank. No mass changes.
- Hand edit after export: the OSL tibial pylon is 22.5 mm longer. In the stock
  export the prosthetic leg sits 22.5 mm shorter than the intact leg (vertical
  hip to lowest sole collision point, 868.3 vs 845.8 mm at qpos0), almost all of
  it below the knee. `OSL_KA_L1_osl_ankle_assembly` moves from y = -0.1561 to
  -0.1786, and the pylon cylinder and inertial grow at the distal end. The
  length was chosen with `myoassist/models/osl_shank_extension.py`. Real OSL
  pylons are cut to length, so this fits the device to the leg rather than
  changing its design. Total mass rises by 10.9 g to 82.28906 kg. The comment
  block at the top of each vendored XML is the source's own record of the edit.
- Meshes: 70 from [myo_sim](https://github.com/MyoHub/myo_sim)
  (`myo_sim/models/meshes/`), 6 OSL parts from `assist_sim`
  (`assist_sim/models/OpenSourceLeg/mesh/`). 76 unique STLs, about 22 MB. myo_sim
  0.2.3 changed no meshes, and all 76 are byte-identical to the set v2 vendored.
- Scene: `musclemimic_scene.xml` and `musclemimic_skybox.png`, the same files
  every musclemimic full-body model uses. They are tracked here and not
  regenerated.

The `assist_sim` exporter rewrites mesh paths relative to the output file and
strips `meshdir`, so re-exporting straight into this repo does not make the
model portable. It would point back at the exporting machine's `myo_sim`
checkout and virtualenv. The vendoring script is what makes it standalone.

## Edits applied when vendoring

All mechanical, all in `scripts/vendor_osl_fullbody_model.py`:

1. The 108 `file=` mesh attributes are reduced to bare basenames, with
   `meshdir="../meshes"` added to the existing `<compiler>`. `texturedir` is
   deliberately not set, because the scene references its skybox bare and
   MuJoCo resolves it against the including file's directory.
2. The export's builtin `<texture name="sky" builtin="gradient">` is deleted.
   The scene declares a file-backed texture of the same name, and duplicate
   texture names are a hard compile error.
3. The exporter's `<visual>` boilerplate is deleted, so the scene's headlight,
   haze, and camera values win.
4. `<include file="../scene/musclemimic_scene.xml"/>` adds the floor, light,
   and skybox the export does not ship, and the model is renamed `OSLFullBody`.

Contacts, contype and conaffinity, keyframes, actuators, sensors, defaults, and
every inertial value are the source byte for byte.

## The two socket variants

`body/osl_fullbody.xml` is the official rigid weld: nq 120, nv 119. Its worst
`dof_damping * dt / M_ii` at the 2 ms training timestep is 0.995 (a finger
joint), inside the explicit-Euler bound of 2.0, so Euler damping may be
disabled as in plain MyoFullBody.

`body/osl_fullbody_compliant_socket.xml` adds four compliant socket DoFs,
`socket_piston` and `socket_rotation_1..3`, taken from the hand-assembled model:
nq 124, nv 123, same mass. Its worst ratio is 15.68 (`socket_rotation_3`), and
explicit Euler raises `mjWARN_BADQACC` on the first step from key `stand` unless Euler
damping stays implicit. The transplant is exact: in both models the socket body
is a direct child of `femur_r` with no `quat` and an identical `<inertial>`, and
all four joints sit at `pos="0 0 0"`, so the axis vectors carry over verbatim.
`armature="0.01"` is stated explicitly because the official socket body
inherits no `myolegs` default class.

The socket choice and `mjDSBL_EULERDAMP` are therefore one coupled toggle, and
`OSLFullBody(compliant_socket=...)` sets both. The instability belongs to the
socket, not to the prosthesis.

There are two files rather than one file plus a runtime toggle because the
export ships five keyframes, and deleting a joint from an `MjSpec` that has
keyframes fails to compile (`keyframe 'stand': invalid qpos size`). Adding a
joint is safe; deleting one is not.

`body/osl_fullbody_robot.xml` and `body/osl_fullbody_robot_compliant_socket.xml`
are the position-controlled robot twins of the two variants, with the fingers
removed and every muscle replaced by a normalized position servo.

## Quick load test

Load by absolute path. MuJoCo 3.4 resolves nested include and texture paths
against the main file's directory, so a relative path fails to find the skybox.

```bash
.venv/bin/python -c "
import mujoco, os
p = os.path.abspath('models/osl_fullbody/body/osl_fullbody.xml')
m = mujoco.MjModel.from_xml_path(p)
print(m.nq, m.nv, m.nu, m.nbody, m.nkey, m.body_mass.sum())"
# expected: 120 119 403 106 5 82.28905948963704
```

## Trajectories, and the clipper pairing this model depends on

Training clips for this model come from the sibling `clipper` repo
(`/home/dmma/src/00_AMBER/00_projbackflip/clipper`), which vendors its own copy
of this model at `assets/msk/osl_fullbody/`. That copy is produced by the same
four edits and differs from `body/osl_fullbody.xml` only in its one-line banner
comment.

The pairing is load-bearing and nothing at training time checks it. Under
`trajectory_cache_type: "sparse"` the training pipeline keeps only `qpos`,
`joint_names`, `jnt_type`, `frequency`, and `split_points` from a clip and
regenerates every velocity and site position by replaying `mj_forward` on this
model (`loco_mujoco/trajectory/handler.py`). The reference the policy chases is
defined jointly by the clip's `qpos` and this model's geometry. If the two drift
apart, nothing raises and the reference quietly becomes wrong.

If `scripts/vendor_osl_fullbody_model.py` is ever re-run against a changed
source, re-vendor the clipper copy and then regenerate the clips, in that order.

Two unit tests in `tests/unit/test_osl_fullbody_env.py` guard the pairing and
skip when the clip is absent. One checks that forward kinematics here reproduces
the clip's recorded `site_xpos` to within 1e-6 m (float32 rounding is about
1e-7 m). The other compares the clip's recorded `body_pos` against this model's,
which catches a pylon-length mismatch directly.

Clips are generated against the rigid variant, so a clip names 74 joints (the
40 finger joints are removed) and never names the four socket DoFs.
`OSLFullBody` pads those to 0.0, the neutral socket, which is exactly the rigid
configuration. Only those four are exempt from the env's joint name-coverage
gate.

## Licensing

All three sources are Apache-2.0: myoassist, `assist_sim`, and `myo_sim`. The
underlying musculoskeletal model derives from Rajagopal 2016 via MyoConverter;
the OSL geometry is the Open-Source Leg project.
