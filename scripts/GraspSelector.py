'''
Copyright, 2022, Matt Corsaro, matthew_corsaro@brown.edu
'''

import copy
import numpy as np
import open3d as o3d
import os
import random
from scipy.spatial.transform import Rotation
import sys

import argparse 

import pickle

from GraspClassifier import GraspClassifier
import grasp_pose_generator as gpg

# mujoco to pybullet
def wxyz2xyzw(wxyz):
    l_wxyz = list(wxyz)
    return l_wxyz[1:] + [l_wxyz[0]]
# pybullet to mujoco
def xyzw2wxyz(xyzw):
    l_xyzw = list(xyzw)
    return [l_xyzw[-1]] + list(l_xyzw[:-1])

"""
Generates quaternion list (wxyz) from numpy rotation matrix

@param np_rot_mat: 3x3 rotation matrix as numpy array

@return quat: w-x-y-z quaternion rotation list
"""
def mat2Quat(np_rot_mat):
    rot = Rotation.from_matrix(np_rot_mat)
    quat_xyzw = rot.as_quat()
    quat_wxyz = [quat_xyzw[3]] + list(quat_xyzw)[:3]
    return quat_wxyz

"""
Generates position list and quaternion list (wxyz) from numpy transformation matrix

@param np_mat: 4x4 transformation matrix as numpy array

@return pos:  x-y-z position list
@return quat: w-x-y-z quaternion rotation list
"""
def mat2PosQuat(np_mat):
    pos = list(np_mat[:3,3])
    quat_wxyz = mat2Quat(np_mat[:3,:3])
    return (pos, quat_wxyz)

class GraspSelector(object):
    def __init__(self, object_frame, point_cloud_with_normals):
        super(GraspSelector, self).__init__()
        
        # Initializations
        self.object_frame_in_world_frame = object_frame
        self.point_cloud_with_normals = point_cloud_with_normals
        # Distance from sampled point in cloud to designated ee frame
        #self.dist_from_point_to_ee_link = -0.02
        self.dist_from_point_to_ee_link = -0.01

        # Gaussian Process classifier wrapper
        self.clf = GraspClassifier()


        self.generateGraspPosesWorldFrame(point_cloud_with_normals)

    def visualizeGraspPoses(self, grasp_poses):
        # Given a 4x4 transformation matrix, create coordinate frame mesh at the pose
        #     and scale down.
        def o3dTFAtPose(pose, scale_down=10):
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
            scaling_maxtrix = np.ones((4,4))
            scaling_maxtrix[:3, :3] = scaling_maxtrix[:3, :3]/scale_down
            scaled_pose = pose*scaling_maxtrix
            axes.transform(scaled_pose)
            return axes
        world_frame_axes = o3dTFAtPose(np.eye(4))
        models = [world_frame_axes, self.point_cloud_with_normals]
        for grasp_pose in grasp_poses:
            grasp_axes = o3dTFAtPose(grasp_pose, scale_down=100)
            models.append(grasp_axes)
  
        o3d.visualization.draw_geometries(models)
    """
    Initialize the grasp pose generator and generate a grasp pose at each point
        in the provided cloud in the world frame.

    @param point_cloud_with_normals: o3d point cloud with estimated normals

    """
    def generateGraspPosesWorldFrame(self, point_cloud_with_normals):
        self.grasp_generator = gpg.GraspPoseGenerator(point_cloud_with_normals, rotation_values_about_approach=[0])
        self.grasp_poses = []
        for i in range(np.asarray(point_cloud_with_normals.points).shape[0]):
            self.grasp_poses += self.grasp_generator.proposeGraspPosesAtCloudIndex(i)

    # Transform a grasp pose in the world frame to the object frame (used in training.)
    def graspPoseWorldFrameToObjFrame(self, grasp_pose):
        return np.matmul(np.linalg.inv(self.object_frame_in_world_frame), grasp_pose)

    # Transform a grasp pose in the world frame to a list of len(7) representing
    #     the pose in the object frame (position and quaternion)
    def graspPoseWorldFrameToClassifierInput(self, grasp_pose):
        # World frame to object frame
        grasp_pose_in_obj_frame = self.graspPoseWorldFrameToObjFrame(grasp_pose)
        # 4x4 matrix to position, quaternion
        grasp_pose_obj_frame_pos, grasp_pose_obj_frame_quat = mat2PosQuat(grasp_pose_in_obj_frame)
        # [3], [4] to [7]
        example_vector = list(grasp_pose_obj_frame_pos) + list(grasp_pose_obj_frame_quat)
        return example_vector

    """
    Add a new example to the classifier's training set, retrain the classifier

    @param grasp_pose_mat: (4x4) numpy array representing grasp pose, with z-axis
        being the approach direction.
    """
    def updateClassifier(self, grasp_pose_mat, label):
        # grasp pose to list of len(7) representing pose in the object frame.
        example_vector = self.graspPoseWorldFrameToClassifierInput(grasp_pose_mat)
        print("Re-training classifier with additional example {} and label {}.".format(example_vector, label))
        self.clf.addBinaryLabeledExample(example_vector, label)
        self.clf.trainClassifier()

    def getRankedGraspPoses(self):
        # If the classifier has yet to be trained, return all poses in default order
        if self.clf.clf is None:
            return copy.deepcopy(self.grasp_poses)
        else:
            grasp_poses_classifier_input = [self.graspPoseWorldFrameToClassifierInput(pose) for pose in self.grasp_poses]
            scores = self.clf.predictSuccessProbabilities(grasp_poses_classifier_input)
            # Indices corresponding to scores sorted from largest to smallest
            sorted_grasp_indices = scores.argsort()[::-1]
            sorted_grasp_poses = [self.grasp_poses[i] for i in sorted_grasp_indices]
            print("Top grasp has a success probability of", scores[sorted_grasp_indices[0]])
            return sorted_grasp_poses

def test(task, num_samples):
    visualize = False

    test_cloud_file = "./pointclouds/"+task+".ply"
    test_cloud_with_normals = o3d.io.read_point_cloud(test_cloud_file)
    world_frame_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    if visualize:
        o3d.visualization.draw_geometries([test_cloud_with_normals, world_frame_axes])
    assert(np.asarray(test_cloud_with_normals.normals).shape == np.asarray(test_cloud_with_normals.points).shape)

    # Fixed pose for the door handle - if the object moves between episodes,
    #     the code that handles this should be changed, and a new pose should
    #     be passed in at train time.
    door_frame = np.eye(4)
    door_frame[:3, :3] = np.array([\
                                    [0, 0, 1], \
                                    [0, 1, 0], \
                                    [-1, 0, 0]])
    door_frame[:3, 3] = [0.14, 0.348, 0.415]

    """
    if visualize:
        # Visualizes object in both world and object frame
        test_cloud_in_object_frame = copy.deepcopy(test_cloud_with_normals)
        test_cloud_in_object_frame.transform(np.linalg.inv(door_frame))
        shrink_tf = np.eye(4)/10
        shrink_tf[3, 3] = 1
        world_frame_axes.transform(shrink_tf)
        o3d.visualization.draw_geometries([world_frame_axes, test_cloud_in_object_frame, test_cloud_with_normals])
    """

    # test_cloud_with_normals is a point cloud only containing the graspable
    #     area of the door handle, represented in the world frame.
    gs = GraspSelector(door_frame, test_cloud_with_normals)


    # GraspSelector generates a set of grasp poses using the point cloud.
    # This function returns all of these poses (4x4 arrays) ranked from most
    #     likely to succeed to least likely.
    sampled_poses = gs.getRankedGraspPoses()
    #Shuffle poses
    random.shuffle(sampled_poses)
    desired_sampled_poses = sampled_poses[:num_samples]
    desired_sampled_poses = [gpg.translateFrameNegativeZ(p, gs.dist_from_point_to_ee_link) for p in desired_sampled_poses]

    # Here, execute the first feasible grasp in the list and generate a
    #     binary success label.

    # Grasp pose represented as a position and wxyz (Mujoco order) quaternion.
    # The positions here are points from the point cloud - they have to
    #     be transformed to end effector poses before they can be used in
    #     the simulator.
    # Here, you would loop through from beginning to end until you find a
    #     reachable grasp pose.
    """
    top_sampled_pose = sampled_poses[0]
    top_end_effector_pose = gpg.translateFrameNegativeZ(top_sampled_pose, gs.dist_from_point_to_ee_link)
    ee_position, ee_orientation = mat2PosQuat(top_end_effector_pose)
    print(ee_position, ee_orientation)
    """

    pickle.dump(desired_sampled_poses, open("./grasps/"+task+".pkl","wb"))

    gs.visualizeGraspPoses(desired_sampled_poses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task',   required=True, type=str, default="DoorCIP",
                    help='name of task you want grasp poses for')
    parser.add_argument('-n', '--num_samples',   required=True, type=int, default=50,
                help='number of grasp poses to get')


    args = parser.parse_args()
    print("Note that gripper will approach along posive z axis (blue), and align parallel gripper with x axis (red)")
    test(args.task, args.num_samples)