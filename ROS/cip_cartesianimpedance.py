import rospy 
import tf2_ros
from iiwa_msgs.msg import CartesianImpedanceControlMode, CartesianPose, JointVelocity, JointPosition
from iiwa_msgs.srv import ConfigureControlMode
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

import time 
import copy
import numpy as np

CONTROL_FREQ = 100
JOINT_SAFETY_FACTOR = 0.8
MONITORING_SPHERE_RADIUS = 0.135 # meters 
MIN_SPHERE_SEPARATION = 0.1
SPHERE_SIZES = [MONITORING_SPHERE_RADIUS]*7
WORKSPACE_LIMIT_X = [0.3 + MONITORING_SPHERE_RADIUS, np.inf] # TOO CONSERVATIVE? 
WORKSPACE_LIMIT_Y = [0.0 + MONITORING_SPHERE_RADIUS, np.inf] # TODO: opposite for the right arm 
WORKSPACE_LIMIT_Z = [0.75587592 + MONITORING_SPHERE_RADIUS, np.inf] # TODO: revisit with grippers 

# TODOs: 
# - vary sphere sizes per link once the grippers are on 

# MONITORING SPHERES DEFINED HERE 
# https://www.oir.caltech.edu/twiki_oir/pub/Palomar/ZTF/KUKARoboticArmMaterial/KUKA_SunriseOS_111_SI_en.pdf 
# p. 249 

class ArmCommander(object):
    """docstring for ArmCommander"""
    def __init__(self, name):

        if name != "left" and name != "right":
            raise Exception("name = left or right only")
        
        # reference frame of pose commands 
        self.frame = None

        # cartesian pose
        self.x = None
        self.y = None 
        self.z = None
        self.qx = None 
        self.qy = None
        self.qz = None
        self.qw = None 

        # target cartesian pose 
        self.target_x = None
        self.target_y = None
        self.target_z = None
        self.target_qx = None
        self.target_qy = None
        self.target_qz = None
        self.target_qw = None

        # joint data 
        self.qpos = np.zeros(7)
        self.qvel = np.zeros(7)
        self.effort = np.zeros(7)

        # cartesian pose pub 
        self.pub_cart = rospy.Publisher("/iiwa_%s/command/CartesianPose" % name, PoseStamped, queue_size=1)
        self.pub_q = rospy.Publisher("/iiwa_%s/command/JointPosition" % name, JointPosition, queue_size=1)

        # define limits 
        self.safe = True
        self.qpos_lower = JOINT_SAFETY_FACTOR * np.array([-170, -120, -170, -120, -170, -120, -175])*np.pi / 180.0
        self.qpos_upper = -self.qpos_lower
        
        self.effort_lower = JOINT_SAFETY_FACTOR * np.array([-176, -176, -110, -110, -110, -40, -40])
        self.effort_upper = -self.effort_lower

        self.qvel_lower = JOINT_SAFETY_FACTOR * np.array([-98, -98, -100, -130, -140, -180, -180])*np.pi / 180.0
        self.qvel_upper = -self.qvel_lower

        # transform listener
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.link_names = ["iiwa_%s_link_" % name + str(i) for i in range(2,8)] # don't check link 1 since it doesn't translate 
        self.link_names += ["iiwa_%s_link_ee" % name]
        self.link_names = sorted(self.link_names)
        self.link_pos = {}
        
    def cart_callback(self, msg):
        self.frame = msg.poseStamped.header.frame_id
        self.x = msg.poseStamped.pose.position.x
        self.y = msg.poseStamped.pose.position.y
        self.z = msg.poseStamped.pose.position.z

        self.qx = msg.poseStamped.pose.orientation.x
        self.qy = msg.poseStamped.pose.orientation.y
        self.qz = msg.poseStamped.pose.orientation.z
        self.qw = msg.poseStamped.pose.orientation.w

    def set_target(self, xyz, quat):      
        self.target_x = xyz[0]
        self.target_y = xyz[1]
        self.target_z = xyz[2]
        self.target_qx = quat[0]
        self.target_qy = quat[1]
        self.target_qz = quat[2]
        self.target_qw = quat[3]

    def joint_callback(self, msg):
        self.qpos = np.array(msg.position)
        self.effort = np.array(msg.effort)

    def joint_vel_callback(self, msg):
        self.qvel[0] = msg.velocity.a1
        self.qvel[1] = msg.velocity.a2
        self.qvel[2] = msg.velocity.a3
        self.qvel[3] = msg.velocity.a4
        self.qvel[4] = msg.velocity.a5
        self.qvel[5] = msg.velocity.a6
        self.qvel[6] = msg.velocity.a7

    def check_safety(self):
        """ returns True when arm is safe"""

        # check jointspace limits 
        if np.any( self.qpos < self.qpos_lower ) or np.any( self.qpos > self.qpos_upper ):
            self.safe = False
            return False 

        if np.any( self.effort < self.effort_lower ) or np.any( self.effort > self.effort_upper ):
            self.safe = False
            return False 

        if np.any( self.qvel < self.qvel_lower ) or np.any( self.qvel > self.qvel_upper ):
            self.safe = False
            return False 

        # check workspace limits 
        # TODO: account for spheres
        for link_name in self.link_names:

            pos = self.link_pos[link_name]

            if pos[0] < WORKSPACE_LIMIT_X[0] or pos[0] > WORKSPACE_LIMIT_X[1]:
                self.safe = False 
                return False

            if pos[1] < WORKSPACE_LIMIT_Y[0] or pos[1] > WORKSPACE_LIMIT_Y[1]:
                self.safe = False 
                return False

            if pos[2] < WORKSPACE_LIMIT_Z[0] or pos[2] > WORKSPACE_LIMIT_Z[1]:
                self.safe = False 
                return False

        self.last_safe_state = copy.deepcopy(self.qpos)
        return True 

        
    def publish_target_command(self):
 
        if self.target_x is None:
            return 

        if self.unsafe:
            return 

        message = PoseStamped()
        message.header.frame_id = self.frame

        message.pose.position.x = self.target_x
        message.pose.position.y = self.target_y
        message.pose.position.z = self.target_z

        message.pose.orientation.x = self.target_qx
        message.pose.orientation.y = self.target_qy
        message.pose.orientation.z = self.target_qz
        message.pose.orientation.w = self.target_qw
        self.pub_cart.publish(message)
        return 

    def publish_current_command(self): 
        message = PoseStamped()
        message.header.frame_id = self.frame

        message.pose.position.x = self.x
        message.pose.position.y = self.y
        message.pose.position.z = self.z

        message.pose.orientation.x = self.qx
        message.pose.orientation.y = self.qy
        message.pose.orientation.z = self.qz
        message.pose.orientation.w = self.qw
        self.pub_cart.publish(message)
        return 

    def publish_q(self, q):
        message = JointPosition()
        message.position.a1 = q[0]
        message.position.a2 = q[1]
        message.position.a3 = q[2]
        message.position.a4 = q[3]
        message.position.a5 = q[4]
        message.position.a6 = q[5]
        message.position.a7 = q[6]
        self.pub_q.publish(message)

    def update_link_poses(self):

        for link_name in self.link_names:

            got_transform = False
            while not got_transform:
                try:
                    trans = self.tfBuffer.lookup_transform("world", link_name, rospy.Time())
                    self.link_pos[link_name] = np.array([trans.transform.translation.x, 
                                                         trans.transform.translation.y,
                                                         trans.transform.translation.z])
                    got_transform = True 
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    rospy.loginfo(e)

class Dorfl(object):
    """docstring for Dorfl"""
    def __init__(self, left_arm_commander, right_arm_commander):
        super(Dorfl, self).__init__()
        
        self.left = left_arm_commander
        self.right = right_arm_commander

        # precompute pairwise monitoring sphere sizes 
        self.radii = np.zeros((7,7))
        for i in range(7):
            for j in range(7):
                self.radii[i,j] = SPHERE_SIZES[i] + SPHERE_SIZES[j]

    def check_safety(self):
        
        # check arms individually 
        if (not self.left.check_safety()) or (not self.right.check_safety()):
            return False 

        # check pairwise spheres 
        left_pos = np.array([ self.left.link_pos[link_name] for link_name in self.left.link_names ]) # (7 x 3)
        right_pos = np.array([ self.right.link_pos[link_name] for link_name in self.right.link_names ]) # (7 x 3)
        distances = np.linalg.norm(left_pos[:, None, :] - right_pos[None, :, :], axis=-1) # (7 x 7) distance matrix 
        sphere_distances = distances - self.radii
        if np.any(sphere_distances < MIN_SPHERE_SEPARATION):
            return False 

        return True 

        
if __name__ == '__main__':
    
    rospy.init_node('arm_commander', anonymous=True)    

    left_arm = ArmCommander('left')
    right_arm = ArmCommander('right')
    dorfl = Dorfl(left_arm, right_arm)

    rospy.Subscriber("/iiwa_left/state/CartesianPose", CartesianPose, left_arm.cart_callback)
    rospy.Subscriber("/iiwa_left/joint_states", JointState, left_arm.joint_callback)
    rospy.Subscriber("/iiwa_left/state/JointVelocity", JointVelocity, left_arm.joint_vel_callback)

    rospy.Subscriber("/iiwa_right/state/CartesianPose", CartesianPose, right_arm.cart_callback)
    rospy.Subscriber("/iiwa_right/joint_states", JointState, right_arm.joint_callback)
    rospy.Subscriber("/iiwa_right/state/JointVelocity", JointVelocity, right_arm.joint_vel_callback)

    last_safe_state = None
    target_q = np.array([0, 0, 0, 10, 160, 0, 45]) * np.pi / 180. 

    rate = rospy.Rate(CONTROL_FREQ)
    while not rospy.is_shutdown():

        left_arm.update_link_poses()
        rospy.loginfo(left_arm.check_safety())

        # if ArmCommander.check_safety():
        #     ArmCommander.publish_q(target_q)
        # else:
        #     if last_safe_state is None:
        #         last_safe_state = copy.deepcopy(ArmCommander.qpos)

        #     ArmCommander.publish_q(last_safe_state)
        #     rospy.loginfo(ArmCommander.qpos)
        rate.sleep()