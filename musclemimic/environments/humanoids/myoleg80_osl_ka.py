"""
MyoLeg80 OSL_KA environment - 80-muscle transfemoral OSL prosthetic model.

Imported from the myoassist project (myoLeg80_OSL_KA): biological left leg + OSL
prosthetic right leg + single rigid passive torso. No upper-body actuation, no
head, no arms. Reuses the MyoFullBody pipeline by overriding the mimic-site map
to a lower-body-only set so the existing MimicReward works unmodified.
"""

from pathlib import Path

from loco_mujoco.core import ObservationType
from loco_mujoco.core.utils import info_property
from mujoco import MjSpec

import musclemimic
from .myofullbody import MyoFullBody

_REPO_ROOT = Path(musclemimic.__file__).resolve().parent.parent
_MODEL_XML = _REPO_ROOT / "models" / "myoLeg80_OSL_KA" / "myolegs_OSL_KA.xml"


class MyoLeg80_OSL_KA(MyoFullBody):
    """
    80-muscle transfemoral OSL prosthetic model.

    - Biological left leg (40 muscles, hip→knee→ankle→toe joints)
    - OSL prosthetic right leg (14 hip muscles + osl_knee + osl_ankle motors)
    - Rigid passive torso (no head, no arms)

    Tracking is restricted to lower-body mimic sites (pelvis + bilateral
    hip/knee/ankle/toe). On the right side the knee/ankle/toe sites are attached
    to the OSL prosthetic bodies (osl_knee_assembly / osl_ankle_assembly /
    osl_foot_assembly) so retargeted SMPL trajectories still align spatially.
    """

    mjx_enabled = False

    def __init__(self, disable_fingers: bool = False, **kwargs) -> None:
        # OSL_KA has no fingers; flip the parent's default to make intent explicit.
        super().__init__(disable_fingers=disable_fingers, **kwargs)

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        return _MODEL_XML.as_posix()

    @info_property
    def root_body_name(self) -> str:
        return "pelvis"

    @info_property
    def upper_body_xml_name(self) -> str:
        # Single rigid torso body — used by GoalTrajMimic for goal-frame projection.
        return "torso"

    @info_property
    def body2sites_for_mimic(self) -> dict[str, str]:
        return {
            "pelvis": "pelvis_mimic",
            # Biological left leg
            "femur_l": "left_hip_mimic",
            "tibia_l": "left_knee_mimic",
            "talus_l": "left_ankle_mimic",
            "toes_l": "left_toes_mimic",
            # Right leg: biological femur + OSL prosthetic from knee down.
            # OSL bodies stand in for tibia_r/talus_r/toes_r so the retargeted
            # site positions (which are biological in source) still pair with a
            # spatially-consistent point on the prosthetic chain.
            "femur_r": "right_hip_mimic",
            "osl_knee_assembly": "right_knee_mimic",
            "osl_ankle_assembly": "right_ankle_mimic",
            "osl_foot_assembly": "right_toes_mimic",
        }

    def _get_observation_specification(self, spec: MjSpec) -> list[ObservationType]:
        # Reuse parent's joint + muscle observation construction by temporarily
        # disabling its hardcoded touch-sensor block (which references
        # r_foot/r_toes — absent on the OSL prosthetic side), then append the
        # OSL_KA-specific touch sensors.
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


class MjxMyoLeg80_OSL_KA(MyoLeg80_OSL_KA):
    """MJX-enabled variant of MyoLeg80_OSL_KA."""

    mjx_enabled = True

    def __init__(
        self,
        timestep: float = 0.002,
        n_substeps: int = 5,
        mjx_backend: str = "jax",
        **kwargs,
    ):
        # Mirror MjxMyoFullBody: extract goal-related kwargs so they're not
        # forwarded into the viewer, set default mjx model_option_conf.
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
            # Unlike MjxMyoFullBody, keep Euler damping enabled (no
            # mjDSBL_EULERDAMP). The OSL socket_piston joint
            # (stiffness=20000, damping=10000) is past the explicit-Euler
            # stability bound at timestep=0.002 under Warp.
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
