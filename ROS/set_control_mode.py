import rospy 
from iiwa_msgs.msg import CartesianImpedanceControlMode, CartesianPose
from iiwa_msgs.srv import ConfigureControlMode
from geometry_msgs.msg import PoseStamped

rospy.init_node('cart_mode', anonymous=True)
rospy.wait_for_service("/iiwa_left/configuration/ConfigureControlMode")
try:
    imp = CartesianImpedanceControlMode()
    imp.cartesian_stiffness.x = 150.0
    imp.cartesian_stiffness.y = 150.0
    imp.cartesian_stiffness.z = 150.0
    imp.cartesian_stiffness.a = 150.0
    imp.cartesian_stiffness.b = 150.0
    imp.cartesian_stiffness.c = 150.0

    imp.cartesian_damping.x = 0.5
    imp.cartesian_damping.y = 0.5
    imp.cartesian_damping.z = 0.5
    imp.cartesian_damping.a = 0.5
    imp.cartesian_damping.b = 0.5
    imp.cartesian_damping.c = 0.5

    imp.nullspace_stiffness = 100.0
    imp.nullspace_damping = 0.7

    service = rospy.ServiceProxy("/iiwa_left/configuration/ConfigureControlMode", ConfigureControlMode)

    print('calling service...')
    result = service(control_mode=2, 
                     cartesian_impedance=imp)
    print('success')
    rospy.loginfo(result)

except rospy.ServiceException as e:
    rospy.loginfo("Service call failed: %e"%e)
