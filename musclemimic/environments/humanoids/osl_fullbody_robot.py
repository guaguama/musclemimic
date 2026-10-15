"""Position-controlled robot twin of the MyoAssist OSL full-body model (v3).

The standalone models preserve :class:`OSLFullBody`'s fingerless skeleton and
prosthesis but replace every muscle-tendon actuator with a normalized absolute
position servo.  The OSL knee and ankle stay torque controlled.

Action order (29 dimensions)
----------------------------
0--2: torso ``flex_extension``, ``lat_bending``, ``axial_rotation``;
3--9: right shoulder/arm/wrist; 10--16: the corresponding left arm coordinates;
17--19: residual right hip; 20--26: intact left hip/knee/ankle/subtalar/MTP;
27: OSL knee torque; 28: OSL ankle torque.

Body actions have actuator ``ctrlrange=[-1, 1]`` and are passed through by
``DefaultControl``.  The affine servo force is
``Kp * (midpoint + half_range * action - q) - Kd * qdot``, so ``-1`` and ``+1``
target the joint's limits.  The final two normalized actions are scaled by
``DefaultControl`` to the OSL motors' ``[-2.88, 2.88]`` control ranges.

Both socket variants share ONE servo table
------------------------------------------
The sizing analysis is run once, on the rigid model.  That is a deliberate choice
and it is safe rather than exact: the compliant socket sits in the right-leg chain
and lowers effective inertia there by up to 50% (``hip_adduction_r`` 0.3377 ->
0.1648), so the shared ``Kd`` lands at ``zeta ~ 1.43`` on that joint instead of 1.
Overdamped, not unstable -- measured worst ``Kd * dt / I`` is 0.600 on BOTH
variants (``ankle_angle_l``) at the 2 ms training timestep, against an
explicit-Euler bound of 2.0.  On the compliant model the three right-hip
coordinates sit at zeta 1.108 / 1.432 / 1.079 and ``hip_adduction_l`` at 1.042;
every other joint stays within 1%, as does every joint of the rigid model.
"""

from pathlib import Path

from mujoco import MjSpec

import musclemimic

from .osl_fullbody import MjxOSLFullBody, OSLFullBody, _TOUCH_CONTACT_GEOMS


_MODEL_ROOT = (
    Path(musclemimic.__file__).resolve().parent.parent
    / "models"
    / "osl_fullbody"
    / "body"
)
_RIGID_ROBOT_XML = _MODEL_ROOT / "osl_fullbody_robot.xml"
_COMPLIANT_ROBOT_XML = _MODEL_ROOT / "osl_fullbody_robot_compliant_socket.xml"


class OSLFullBodyRobot(OSLFullBody):
    """CPU environment for the muscle-free, position-controlled official OSL robot.

    Inherits the ``compliant_socket`` toggle, the mimic-site map, the touch-sensor
    list and the trajectory name-coverage gate (including its socket-DoF exemption)
    unchanged -- only the pair of source models differs.

    ``disable_fingers`` stays meaningful in an inherited config even though these
    models have no finger joints left: ``_delete_fingers`` filters by name, so it is
    a no-op here rather than an error.
    """

    mjx_enabled = False

    @classmethod
    def _variant_xml(cls, compliant_socket: bool) -> str:
        return (
            _COMPLIANT_ROBOT_XML if compliant_socket else _RIGID_ROBOT_XML
        ).as_posix()

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        return _RIGID_ROBOT_XML.as_posix()


class MjxOSLFullBodyRobot(OSLFullBodyRobot, MjxOSLFullBody):
    """MJX robot variant, retaining the parent's socket-stability model options."""

    mjx_enabled = True

    def _modify_spec_for_mjx(self, spec: MjSpec) -> MjSpec:
        spec = super()._modify_spec_for_mjx(spec)
        if self.mjx_backend != "warp":
            # The parent already re-enables contype on the eight foot geoms after
            # disabling non-ground contacts.  It runs against the muscle model's
            # geom set, which is identical here, so nothing extra is needed --
            # this override exists only to assert that invariant in one place.
            present = {geom.name for geom in spec.geoms}
            missing = set(_TOUCH_CONTACT_GEOMS) - present
            if missing:
                raise RuntimeError(
                    f"robot model is missing touch-sensor geoms: {sorted(missing)}"
                )
        return spec
