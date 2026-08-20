"""Position-controlled OSL full-body robot environments.

The standalone model preserves the fingerless :class:`OSLFullBody` skeleton and
prosthesis, but replaces all muscle-tendon actuators with normalized absolute
position servos.  The OSL knee and ankle remain torque controlled.

Action order (29 dimensions)
----------------------------
0--2: torso ``flex_extension``, ``lat_bending``, ``axial_rotation``;
3--9: right shoulder/arm/wrist; 10--16: corresponding left arm coordinates;
17--19: residual right hip; 20--26: intact left hip/knee/ankle/subtalar/MTP;
27: OSL knee torque; 28: OSL ankle torque.

Body actions have actuator ``ctrlrange=[-1, 1]`` and are passed through by
``DefaultControl``.  Its affine servo force is equivalent to
``Kp * (midpoint + half_range * action - q) - Kd * qdot``, so ``-1`` and ``+1``
target the corresponding joint limits.  The final two
normalized actions are scaled by ``DefaultControl`` to the OSL motors' original
``[-2.88, 2.88]`` control ranges.
"""

from pathlib import Path

from mujoco import MjSpec

import musclemimic

from .osl_fullbody import MjxOSLFullBody, OSLFullBody


_REPO_ROOT = Path(musclemimic.__file__).resolve().parent.parent
_MODEL_XML = _REPO_ROOT / "models" / "osl_fullbody" / "body" / "osl_fullbody_robot.xml"


class OSLFullBodyRobot(OSLFullBody):
    """CPU environment for the muscle-free, position-controlled OSL robot."""

    mjx_enabled = False

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        return _MODEL_XML.as_posix()


class MjxOSLFullBodyRobot(OSLFullBodyRobot, MjxOSLFullBody):
    """MJX robot variant retaining OSL socket-stability model options."""

    mjx_enabled = True

    def _modify_spec_for_mjx(self, spec: MjSpec) -> MjSpec:
        spec = super()._modify_spec_for_mjx(spec)
        if self.mjx_backend != "warp":
            # The CPU source has explicit arm/prosthesis collision pairs.  The
            # JAX MJX backend does not implement every ellipsoid/mesh pairing;
            # parent behavior already disables non-ground geom contacts, so
            # remove the explicit pairs as well for the same no-self-contact
            # JAX approximation.  Warp retains the complete contact model.
            for pair in list(spec.pairs):
                spec.delete(pair)
            # Keep the three touch-sensor regions collidable with the floor.
            # Otherwise MJX has no candidate contacts and its touch-sensor
            # evaluation cannot concatenate contact forces.
            touch_contact_geoms = {
                "osl_foot_col1",
                "osl_foot_col2",
                "osl_foot_col3",
                "l_foot_col1",
                "l_foot_col3",
                "l_foot_col4",
                "l_bofoot_col1",
                "l_bofoot_col2",
            }
            for geom in spec.geoms:
                if geom.name in touch_contact_geoms:
                    geom.contype = 1
        return spec
