from .myofullbody import MyoFullBody, MjxMyoFullBody
from .bimanual import MyoBimanualArm, MjxMyoBimanualArm
from .osl_fullbody import OSLFullBody, MjxOSLFullBody
from .osl_fullbody_robot import OSLFullBodyRobot, MjxOSLFullBodyRobot


# register muscle environments
MyoBimanualArm.register()
MjxMyoBimanualArm.register()
MyoFullBody.register()
MjxMyoFullBody.register()
OSLFullBody.register()
MjxOSLFullBody.register()
OSLFullBodyRobot.register()
MjxOSLFullBodyRobot.register()
