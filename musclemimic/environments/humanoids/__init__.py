from .myofullbody import MyoFullBody, MjxMyoFullBody
from .bimanual import MyoBimanualArm, MjxMyoBimanualArm
from .myoleg80_osl_ka import MyoLeg80_OSL_KA, MjxMyoLeg80_OSL_KA
from .osl_fullbody import OSLFullBody, MjxOSLFullBody
from .osl_fullbody_robot import OSLFullBodyRobot, MjxOSLFullBodyRobot


# register muscle environments
MyoBimanualArm.register()
MjxMyoBimanualArm.register()
MyoFullBody.register()
MjxMyoFullBody.register()
MyoLeg80_OSL_KA.register()
MjxMyoLeg80_OSL_KA.register()
OSLFullBody.register()
MjxOSLFullBody.register()
OSLFullBodyRobot.register()
MjxOSLFullBodyRobot.register()
