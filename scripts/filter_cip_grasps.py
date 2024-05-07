import os
import time
import argparse 
import sys 
import pickle
from copy import deepcopy

import numpy as np
import torch
import gym 
import matplotlib.pyplot as plt

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.cip_env import GRIP_NAMES

from baselines_robosuite.baselines_wrapper import BaselinesWrapper


"""
grip_names

DoorCIP - Door_handle
DrawerCIP - Drawer_handle
SlideCIP - Slide_grip
LeverCIP - Lever_lever
"""

# TODO: SEEDING esp for random door pose init. 
# TODO: reset object to consistent pose for grasp evaluation?! 

RENDER = False
NUM_QPOS_PER_GRASP = 1
NUM_ATTEMPTS_PER_GRASP = 10

ground_geom_id = None
robot_geom_ids = []
obj_geom_ids = []

def setGeomIDs(env):
    global ground_geom_id
    global robot_geom_ids
    global obj_geom_ids

    for n in range(env.sim.model.ngeom):
        body = env.sim.model.geom_bodyid[n]
        body_name = env.sim.model.body_id2name(body)
        geom_name = env.sim.model.geom_id2name(n)

        if geom_name == "ground" and body_name == "world":
            ground_geom_id = n
        elif "robot0_" in body_name or "gripper0_" in body_name:
            robot_geom_ids.append(n)
        elif body_name != "world":
            print(geom_name)
            obj_geom_ids.append(n)

def contactBetweenRobotAndObj(contact):
    global ground_geom_id
    global robot_geom_ids
    global obj_geom_ids
    if contact.geom1 in robot_geom_ids and contact.geom2 in obj_geom_ids:
        print("Contact between {one} and {two}".format(one=contact.geom1, two=contact.geom2))
        return True
    if contact.geom2 in robot_geom_ids and contact.geom1 in obj_geom_ids:
        print("Contact between {one} and {two}".format(one=contact.geom1, two=contact.geom2))
        return True
    return False

def contactBetweenGripperAndSpecificObj(contact, name):
    global ground_geom_id
    global robot_geom_ids
    global obj_geom_ids

    if env.sim.model.geom_id2name(contact.geom1)[:8] == 'gripper0' and env.sim.model.geom_id2name(contact.geom2) == name:
        # print("Contact between {one} and {two}".format(one=env.sim.model.geom_id2name(contact.geom1), two=env.sim.model.geom_id2name(contact.geom2)))
        return True
    if env.sim.model.geom_id2name(contact.geom2)[:8] == 'gripper0' and env.sim.model.geom_id2name(contact.geom1) == name:
        # print("Contact between {one} and {two}".format(one=env.sim.model.geom_id2name(contact.geom1), two=env.sim.model.geom_id2name(contact.geom2)))
        return True
    return False

def contactBetweenRobotAndFloor(contact):
    global ground_geom_id
    global robot_geom_ids
    global obj_geom_ids

    if contact.geom1 == ground_geom_id and contact.geom2 in robot_geom_ids:
        return True
    if contact.geom2 == ground_geom_id and contact.geom1 in robot_geom_ids:
        return True
    return False

def isInvalidMJ(env):
    # Note that the contact array has more than `ncon` entries,
    # so be careful to only read the valid entries.
    for contact_index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[contact_index]
        if contactBetweenRobotAndObj(contact):
            return 1
        elif contactBetweenRobotAndFloor(contact):
            return 2
    return 0 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task',   required=True, type=str, 
                        help='name of task. DrawerCIP, DoorCIP, ...')
    args = parser.parse_args()
    print(args)

    grip_name = GRIP_NAMES[args.task]

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
    options["ee_fixed_to_handle"] = True
    options["hard_reset"] = False


    # create and wrap env 
    raw_env = suite.make(
        **options,
        has_renderer=RENDER,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
    )
    env = BaselinesWrapper(raw_env,
                           pregrasp_policy=False, 
                           grasp_strategy="fixed", 
                           learning=False,
                           optimal_ik=False,
                           use_cached_qpos=False)
    env.set_render(RENDER)

    heuristic_grasps_path = "./grasps/"+args.task+".pkl"
    heuristic_grasps_filtered_path = "./grasps/"+args.task+"_filtered.pkl"
    heuristic_grasps = pickle.load(open(heuristic_grasps_path,"rb"))
    reference_obj_pose = np.load(f"./pointclouds/{args.task}_pose.npy")

    good_grasps = []
    print("Number of heuristic grasps: {num}".format(num=len(heuristic_grasps)))

    setGeomIDs(env)
    print(f'USING GRIP NAME: {grip_name}')

    wp_list = []

    for try_grasp in heuristic_grasps:
       
        # try_grasp is in world frame of original task env 
        # convert into current world frame 
        # TODO: do this elsewhere, read in GraspSelector and/or put pcd in obj frame 
        try_grasp_obj_frame = np.linalg.inv(reference_obj_pose) @ try_grasp
        env.grasp_pose = try_grasp_obj_frame
        cur_grasp_wp_list = []
        cur_grasp_qpos_list = []
        attmpt_counter = 0 
        is_valid_grasp = False 
        while attmpt_counter < NUM_ATTEMPTS_PER_GRASP:
    
            attmpt_counter += 1 
            env.reset()
            if RENDER: 
                env.render()

            w,p,wp = env.check_manipulability() 

            if not env.grasp_success:
                continue
            
            if RENDER: 
                env.render()
                input("GRASP PASSED IK AND COLLISION CHECK 1")

            grasp_qpos = deepcopy(env.sim.data.qpos[:7])

            #close gripper for a few frames
            a = np.zeros(env.action_spec[0].shape)
            a[-1] = 1
            for _ in range(10):
                o, r, d, i = env.step(a)
                if i["unsafe_qpos"]:
                    break
                if RENDER: env.render()

            #check for joint state violations after
            if env.robots[0].check_q_limits() or i["unsafe_qpos"]:
                print("[X] Joint state violation after pregrasp")
                wp_list.append([wp,0,"blue"])
                continue
            
            '''
            #Do a random walk for 100 frames or something
            a = np.zeros(env.action_spec[0].shape)
            for _ in range(100):
                a = np.random.uniform(-.15, .15, a.shape)
                a[-1] = 1
                o, r, d, i = env.step(a)
            '''          
            
            handle_contact = False
            for contact_index in range(env.sim.data.ncon):
                contact = env.sim.data.contact[contact_index]
                if contactBetweenGripperAndSpecificObj(contact, grip_name):
                    handle_contact=True

            if not handle_contact:
                print("Lost contact with handle after random walk")
                wp_list.append([wp,0,"orange"])
                continue

            is_valid_grasp = True 
            cur_grasp_qpos_list.append(grasp_qpos)
            cur_grasp_wp_list.append(wp)
            wp_list.append([wp,1,"green"])
            break 

        if not is_valid_grasp: 
            wp_list.append([wp,0,"red"])
            if RENDER: input("GRASP IK INFEASIBLE OR IN COLLISION")
            continue

        good_grasps.append([try_grasp_obj_frame, cur_grasp_wp_list, cur_grasp_qpos_list])

    pickle.dump(good_grasps, open(heuristic_grasps_filtered_path,"wb"))
    wp_vals, wp_results, wp_colors = list(zip(*wp_list))

    #red: IK infeasible or in collision
    #blue: joint violation after closing gripper
    #orange: lost contact after gripping
    #green: all good!
    print("Total successful grasps:{good_grasps}".format(good_grasps=float(len(good_grasps))))
    print("Fraction of successful grasps:{frac}".format(frac=float(len(good_grasps))/len(heuristic_grasps)))
    if RENDER:
        plt.scatter(wp_vals,wp_results,c=wp_colors)
        plt.show()

