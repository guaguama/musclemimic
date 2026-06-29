"""
OSL_FullBody environment - full-body muscle humanoid with the OSL knee-ankle prosthesis.

This is the MyoFullBody model with the right leg amputated transfemorally and the OSL
prosthesis (from myoLeg80_OSL_KA) grafted on: biological left leg, residual right femur,
OSL prosthetic right shank/foot (osl_knee + osl_ankle motors), plus the full upper body
(torso, head, both arms) with muscle actuation.

Reuses the MyoFullBody pipeline by overriding two things that the parent assumes about a
fully-biological skeleton:
  - the mimic-site map: the right knee/ankle/toes sites attach to the OSL prosthetic
    bodies (osl_knee_assembly / osl_ankle_assembly / osl_foot_assembly) because the
    biological tibia_r / talus_r / toes_r bodies were amputated.
  - the touch sensors: the right biological foot/toe sensors (r_foot / r_toes) are absent;
    the OSL side exposes r_osl_foot instead.

The MJX variant keeps Euler damping ENABLED (same fix as MjxMyoLeg80_OSL_KA): the OSL
socket_piston joint (stiffness=20000, damping=10000) diverges to NaN under the Warp
backend at timestep=0.002 with explicit Euler.
"""

from pathlib import Path

from loco_mujoco.core import ObservationType
from loco_mujoco.core.utils import info_property
from mujoco import MjSpec

import musclemimic
from .myofullbody import MyoFullBody

_REPO_ROOT = Path(musclemimic.__file__).resolve().parent.parent
_MODEL_XML = _REPO_ROOT / "models" / "osl_fullbody" / "body" / "osl_fullbody.xml"


class OSLFullBody(MyoFullBody):
    """
    Full-body muscle humanoid with the OSL knee-ankle prosthesis on the right leg.

    Tracking uses the full 17-site full-body mimic set (pelvis + upper body + both arms +
    both legs). On the right side the knee/ankle/toes sites are attached to the OSL
    prosthetic bodies (osl_knee_assembly / osl_ankle_assembly / osl_foot_assembly) so the
    retargeted trajectories (biological in source) still pair with a spatially-consistent
    point on the prosthetic chain.
    """

    mjx_enabled = False

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        return _MODEL_XML.as_posix()

    @info_property
    def body2sites_for_mimic(self) -> dict[str, str]:
        return {
            "pelvis": "pelvis_mimic",
            "lumbar1": "upper_body_mimic",
            "head": "head_mimic",
            # Left arm (note: left uses lowercase l suffix)
            "humerus_l": "left_shoulder_mimic",
            "ulna_l": "left_elbow_mimic",
            "lunate_l": "left_hand_mimic",
            # Right arm
            "humerus_r": "right_shoulder_mimic",
            "ulna_r": "right_elbow_mimic",
            "lunate_r": "right_hand_mimic",
            # Biological left leg
            "femur_l": "left_hip_mimic",
            "tibia_l": "left_knee_mimic",
            "talus_l": "left_ankle_mimic",
            "toes_l": "left_toes_mimic",
            # Right leg: biological residual femur + OSL prosthetic from knee down.
            # OSL bodies stand in for tibia_r/talus_r/toes_r so the retargeted site
            # positions (biological in source) still pair with a spatially-consistent
            # point on the prosthetic chain.
            "femur_r": "right_hip_mimic",
            "osl_knee_assembly": "right_knee_mimic",
            "osl_ankle_assembly": "right_ankle_mimic",
            "osl_foot_assembly": "right_toes_mimic",
        }

    def _get_observation_specification(self, spec: MjSpec) -> list[ObservationType]:
        # Reuse parent's joint + muscle observation construction by temporarily
        # disabling its hardcoded touch-sensor block (which references r_foot/r_toes —
        # absent on the OSL prosthetic side), then append the OSL-specific touch sensors.
        touch_was_enabled = self._enable_touch_sensor_observations
        self._enable_touch_sensor_observations = False
        try:
            obs_spec = super()._get_observation_specification(spec)
        finally:
            self._enable_touch_sensor_observations = touch_was_enabled

        if touch_was_enabled:
            for sensor_name in ("r_osl_foot", "l_foot", "l_toes"):
                obs_spec.append(
                    ObservationType.TouchSensor(f"touch_{sensor_name}", xml_name=sensor_name)
                )
        return obs_spec


class MjxOSLFullBody(OSLFullBody):
    """MJX-enabled variant of OSLFullBody."""

    mjx_enabled = True

    def __init__(
        self,
        timestep: float = 0.002,
        n_substeps: int = 5,
        mjx_backend: str = "jax",
        **kwargs,
    ):
        # Mirror MjxMyoFullBody: extract goal-related kwargs so they're not forwarded into
        # the viewer, set default mjx model_option_conf.
        goal_related_params = [
            "visualize_goal",
            "enable_enhanced_visualization",
            "target_geom_rgba",
            "n_step_lookahead",
            "goal_type",
            "goal_params",
        ]
        extracted_goal_params = {p: kwargs.pop(p) for p in goal_related_params if p in kwargs}

        if "model_option_conf" not in kwargs:
            # Unlike MjxMyoFullBody, keep Euler damping enabled (no mjDSBL_EULERDAMP).
            # The OSL socket_piston joint (stiffness=20000, damping=10000) is past the
            # explicit-Euler stability bound at timestep=0.002 under Warp, and diverges
            # to NaN without implicit Euler damping. This is the same fix that made
            # MjxMyoLeg80_OSL_KA train.
            model_option_conf = dict(
                iterations=4,
                ls_iterations=8,
            )
        else:
            model_option_conf = kwargs.pop("model_option_conf")

        kwargs.update(extracted_goal_params)
        self.mjx_backend = mjx_backend

        super().__init__(
            timestep=timestep,
            n_substeps=n_substeps,
            model_option_conf=model_option_conf,
            mjx_backend=mjx_backend,
            **kwargs,
        )
