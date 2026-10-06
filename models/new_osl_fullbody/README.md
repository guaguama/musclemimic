# new_osl_fullbody (imported)

Self-contained copy of the **official** NeuMoVe Lab full-body + OSL knee-ankle
prosthesis model. This is the upstream replacement for the hand-assembled
`models/osl_fullbody/`, which grafted the `myoLeg80_OSL_KA` prosthetic leg onto
the packaged MyoFullBody model. `models/osl_fullbody/` is deliberately left
untouched; this directory is additive.

Everything here except `README.md` is generated. Run
`scripts/vendor_new_osl_fullbody_model.py` rather than editing the XMLs, and
`--check` to prove the tracked tree matches a deterministic regeneration.

## Provenance

- Upstream repo: [myoassist](https://github.com/neumovelab/myoassist) at commit
  `acee39b7` (2026-08-26), fork `github.com/guaguama/myoassist`.
- The export is produced by the `assist_sim` 1.0.0 CLI, not by myoassist itself:

  ```bash
  cd <myoassist repo>
  python -m assist_sim combine myofullbody OpenSourceLeg_KA_L1 \
      -o models/myofullbody_osl_ka.xml
  ```

  The amputation recipe is `assist_sim/models/OpenSourceLeg/KA_L1config.yaml`
  (body removals, mesh replacements, residual-femur mass override, muscle
  removals and tendon re-anchorings, the two OSL motors, keyframes, sensors).
  The export is untracked in myoassist, so this README is its only record.
- Meshes: 70 from [myo_sim](https://github.com/MyoHub/myo_sim) `0.2.2`
  (`myo_sim/models/meshes/`), 6 OSL parts from `assist_sim`
  (`assist_sim/models/OpenSourceLeg/mesh/`). 76 unique STLs, ~22 MB.
- Scene: `musclemimic_scene.xml` and `musclemimic_skybox.png` copied verbatim
  from `models/osl_fullbody/scene/` so the new model renders identically to the
  existing ones.

`assist_sim`'s exporter rewrites mesh paths relative to the *output file* and
strips `meshdir`, so re-exporting straight into this repo does **not** make the
model portable — it points back at the exporting machine's `myo_sim` checkout
and virtualenv. The vendoring script is what makes it standalone.

## Edits applied to the export

All mechanical, all in `scripts/vendor_new_osl_fullbody_model.py`:

1. The 108 `file=` mesh attributes reduced to bare basenames, with
   `meshdir="../meshes"` added to the existing `<compiler>`. `texturedir` is
   deliberately not set — the scene references its skybox bare and MuJoCo
   resolves it against the *including* file's directory.
2. The export's builtin `<texture name="sky" builtin="gradient">` deleted. Not
   cosmetic: the scene declares a file-backed texture of the same name, and
   duplicate texture names are a hard compile error.
3. The exporter's `<visual>` boilerplate deleted, so the scene's
   headlight/haze/camera values win outright.
4. `<include file="../scene/musclemimic_scene.xml"/>` added for the floor,
   light, and skybox the export does not ship.

Contacts, contype/conaffinity, keyframes, actuators, sensors, defaults, and
every inertial value are the official export byte-for-byte.

## The two variants

| | `new_osl_fullbody.xml` | `new_osl_fullbody_compliant_socket.xml` |
|---|---|---|
| socket | rigidly welded (official) | 4 compliant DoFs |
| nq / nv | 120 / 119 | 124 / 123 |
| mass | 82.27817 kg | 82.27817 kg |
| worst `dof_damping·dt/M_ii` @ 2 ms | 0.995 (a finger joint) | 15.68 (`socket_rotation_3`) |
| explicit Euler @ 2 ms | stable | `mjWARN_BADQACC` at t = 0.016 s |

The compliant variant restores `socket_piston` + `socket_rotation_1..3` from
`models/osl_fullbody/leg/assets/myolegs_chain.xml`. The transplant is exact: in
both models the socket body is a direct child of `femur_r` with no `quat` and an
identical `<inertial>`, and all four joints sit at `pos="0 0 0"`, so the axis
vectors carry over verbatim. `armature="0.01"` is stated explicitly because the
official socket body inherits no `myolegs` default class. The joints add no mass.

**These are a coupled toggle with `mjDSBL_EULERDAMP`.** The explicit-Euler bound
is 2.0. Rigid socket → Euler damping may be disabled, matching plain MyoFullBody.
Compliant socket → Euler damping must stay implicit, which is the workaround the
hand-built OSL environments carry. It belongs to the socket, not the prosthesis.

Two files rather than one file plus a runtime toggle because the export ships
five keyframes, and deleting a joint from an `MjSpec` that has keyframes fails to
compile (`keyframe 'stand': invalid qpos size`). Adding is safe, deleting is not.

## Layout

```
new_osl_fullbody/
├── body/
│   ├── new_osl_fullbody.xml                    # rigid socket (official)
│   └── new_osl_fullbody_compliant_socket.xml   # + 4 socket DoFs
├── meshes/                                     # 76 STLs
└── scene/                                      # floor, light, skybox
```

## Quick load test

Load by **absolute** path. As with `models/osl_fullbody/`, MuJoCo 3.4 resolves
nested include/texture paths against the main file's directory, so a relative
path fails to find the skybox.

```bash
.venv/bin/python -c "
import mujoco, os
p = os.path.abspath('models/new_osl_fullbody/body/new_osl_fullbody.xml')
m = mujoco.MjModel.from_xml_path(p)
print(m.nq, m.nv, m.nu, m.nbody, m.nkey, m.body_mass.sum())"
# expected: 120 119 403 106 5 82.27816858963705
```

## Trajectories, and the clipper pairing this model silently depends on

Training clips for this model are produced by the sibling `clipper` repo
(`/home/dmma/src/00_AMBER/00_projbackflip/clipper`), which **vendors a
byte-identical copy of this directory** at `assets/msk/new_osl_fullbody/`.

That byte-identity is load-bearing and unchecked. Under
`trajectory_cache_type: "sparse"` the training pipeline keeps only `qpos`,
`joint_names`, `jnt_type`, `frequency` and `split_points` from a clip and
regenerates every velocity and site position by replaying `mj_forward` on
*this* model (`loco_mujoco/trajectory/handler.py:442-443`). So the reference the
policy chases is defined jointly by the clip's `qpos` and this model's geometry.
If the two drift apart, **nothing raises** — the reference just becomes quietly
wrong.

> **If `scripts/vendor_new_osl_fullbody_model.py` is ever re-run, re-vendor the
> clipper copy and regenerate the clips.** In that order.

Cheap way to confirm the pairing after any change — forward kinematics on this
model must reproduce the clip's own recorded `site_xpos`:

```bash
.venv/bin/python -c "
import numpy as np, mujoco
from musclemimic.environments.humanoids.new_osl_fullbody import NewOSLFullBody
d = np.load('<clip>.npz', allow_pickle=True)
env = NewOSLFullBody(strict_trajectory_joints=False, horizon=10)
m, dat = env._model, mujoco.MjData(env._model)
ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, str(s)) for s in d['site_names']]
err = []
for f in range(0, len(d['qpos']), 7):
    dat.qpos[:] = d['qpos'][f]; mujoco.mj_forward(m, dat)
    err.append(np.abs(dat.site_xpos[ids] - d['site_xpos'][f]).max())
print('max |d| =', max(err))"
# expected: ~1.2e-07 m (float32 rounding).  Anything at 1e-3 or above means the
# clip was generated against a different model -- regenerate it.
```

This check is also a unit test in `tests/unit/test_new_osl_fullbody_env.py`,
skipped when the clip is not present.

### Clipper-side registration

Adding this model to clipper touched seven entries in `src/clipper/constants.py`
— `NEW_OSL_FULLBODY_XML`, plus registration in `ROBOT_XML`, `MSK_MODELS`,
`RIGHT_ANKLE_JOINT`, `MUSCLEMIMIC_SOURCE_ALIASES` (via a new
`NEW_OSL_FULLBODY_ALIASES`), `MSK_REMOVE_JOINTS`, and `MSK_MIMIC_SITES` — and one
fix in `src/clipper/model.py`, where `_remove_joints` now trims `spec.keys[*].qpos`
before deleting a joint. That fix is required rather than cosmetic: this is the
first MSK model with real `<key>` elements, and deleting a joint from a spec that
has keyframes fails to compile (`keyframe 'stand': invalid qpos size`) — the same
trap documented under "The two variants" above.

Clips are generated with `--pad-standing --ankle-offset -5 --format musclemimic`
against the **rigid** variant. A clip therefore names 74 joints (the 40 finger
joints are removed) and never names the compliant variant's four socket DoFs;
`NewOSLFullBody` pads those to 0.0, which is the neutral socket and hence exactly
the rigid configuration. Only those four are exempt from the env's joint
name-coverage gate.

## Licensing

All three sources are Apache-2.0: myoassist, `assist_sim`, and `myo_sim`. The
underlying musculoskeletal model derives from Rajagopal 2016 via MyoConverter;
the OSL geometry is the Open-Source Leg project.
