import os
import time
import argparse 
import sys 

import numpy as np
import torch
import gym 

import matplotlib.pyplot as plt
plt.plot()
plt.close()

import robosuite as suite
from robosuite.controllers import load_controller_config

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from common import utils, utils_for_q_learning
from baselines_robosuite.baselines_wrapper import BaselinesWrapper, SaveOnBestTrainingRewardCallback


from robosuite.utils import camera_utils
import open3d as o3d

import copy

RENDER = False

#Element ID info
#DoorCIP: handle is 46
#DrawerCIP: handle is 121
#LeverCIP: handle is 51
#SlideCIP: handle is 51

def vertical_flip(img):
    return np.flip(img, axis=0)

def task2handlePC(env, pointcloud_cameras, task, debug,path,element_id):
    obs = env.reset() #Get the raw reset observation from robosuite
    obj_pose = env.get_obj_pose()

    masked_pcd_list = []
    for camera in pointcloud_cameras:
        #vertical flip because OpenGL buffer loading is backwards
        depth_image = vertical_flip(obs[camera+'_depth'])
        rgb_image = vertical_flip(obs[camera+'_image'])
        segmentation_image = vertical_flip(obs[camera+'_segmentation_element'])

        #depth image is normalized by robosuite, this gets real depth map
        depth_image_numpy = camera_utils.get_real_depth_map(env.sim, depth_image)

        if debug:
            f, axarr = plt.subplots(3,1) 
            axarr[0].imshow(rgb_image)
            axarr[1].imshow(depth_image)
            axarr[2].imshow(segmentation_image)
            plt.show()

        #make a mask for the graspable part of object based on element id 
        masked_segmentation = np.where(segmentation_image == int(element_id), 1.0, -1.0)

        #apply masked segmentation to depth image
        masked_depth_image_numpy = np.multiply(masked_segmentation,depth_image_numpy).astype(np.float32)
        #convert to open3d image
        masked_depth_image = o3d.geometry.Image(masked_depth_image_numpy)

        #Get extrinsics of camera
        extrinsic_cam_parameters= camera_utils.get_camera_extrinsic_matrix(env.sim, camera)

        assert depth_image.shape == segmentation_image.shape

        #All images should have same shape, so we can just use depth image for width and height
        img_width = depth_image.shape[1]
        img_height = depth_image.shape[0]

        intrinisc_cam_parameters_numpy = camera_utils.get_camera_intrinsic_matrix(env.sim, camera, img_width, img_height)
        cx = intrinisc_cam_parameters_numpy[0][2]
        cy = intrinisc_cam_parameters_numpy[1][2]
        fx = intrinisc_cam_parameters_numpy[0][0]
        fy = intrinisc_cam_parameters_numpy[1][1]

        intrinisc_cam_parameters = o3d.camera.PinholeCameraIntrinsic(img_width, #width 
                                                            img_height, #height
                                                            fx,
                                                            fy,
                                                            cx,
                                                            cy)

        masked_pcd = o3d.geometry.PointCloud.create_from_depth_image(masked_depth_image,                                                       
                                              intrinisc_cam_parameters
                                             )

        if len(masked_pcd.points) == 0:
            print("Camera {camera} has no masked points, skipping over".format(camera=camera))
            continue

        #Transform the pcd into world frame based on extrinsics
        masked_pcd.transform(extrinsic_cam_parameters)

        #estimate normals
        masked_pcd.estimate_normals()
        #orientation normals to camera
        masked_pcd.orient_normals_towards_camera_location(extrinsic_cam_parameters[:3,3])

        if debug:
            f, axarr = plt.subplots(2,2) 
            axarr[0,0].imshow(rgb_image)
            axarr[1,0].imshow(depth_image)
            axarr[0,1].imshow(segmentation_image)
            axarr[1,1].imshow(masked_segmentation)
            plt.show()

        masked_pcd_list.append(copy.deepcopy(masked_pcd))

    for i in range(len(masked_pcd_list)-1):
        complete_masked_pcd = masked_pcd_list[i] + masked_pcd_list[i+1]
    if (len(masked_pcd_list) == 1):
        complete_masked_pcd = masked_pcd_list[0]

    o3d.io.write_point_cloud(path+task+".ply", complete_masked_pcd)
    np.save(path+task+"_pose.npy", obj_pose)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task',   required=True, type=str, 
                        help='name of task. DrawerCIP, DoorCIP, ...')

    parser.add_argument("--seed", default=0, help="seed",
                        type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--path", default="./pointclouds/", type=str, help="location to save point clouds")
    parser.add_argument("--debug",
                    type=utils.boolify,
                    help="visualize images from cameras?",
                    default=True)
    parser.add_argument('-c','--cameras', nargs='+', default=["birdview","frontview","agentview","sideview"], help='list of cameras to use')
    parser.add_argument("--element_id", default=53, help="element id of object to be segmented")

    args = parser.parse_args()
    print(args)
    pointcloud_cameras = args.cameras

    # TODO: argparse these...
    config = {"training_steps":3000000}

    # create environment instance
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config["impedance_mode"] = "variable_kp"
    controller_config["scale_stiffness"] = True

    # set up env options 
    options = {}
    options["env_name"] = args.task
    options["robots"] = "Panda"
    options["controller_configs"] = controller_config
    options["ee_fixed_to_handle"] = False

    # create and wrap env 
    raw_env = suite.make(
        **options,
        has_renderer=RENDER,
        camera_names=pointcloud_cameras,
        camera_segmentations="element", #element, class, instance
        has_offscreen_renderer=True,
        use_camera_obs=True,
        reward_shaping=True,
        camera_depths=True
    )
    task2handlePC(raw_env, pointcloud_cameras, args.task, args.debug, args.path, args.element_id) #make pointcloud


