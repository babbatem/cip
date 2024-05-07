import os
import time
import argparse 
import sys 

import numpy as np
import torch
import gym 
import pickle

import robosuite as suite
from robosuite.controllers import load_controller_config

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor

from common import utils, utils_for_q_learning
from baselines_robosuite.baselines_wrapper import BaselinesWrapper, SaveOnBestTrainingRewardCallback

from os import listdir


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task',   required=True, type=str, 
                    help='name of task. DrawerCIP, DoorCIP, ...')
parser.add_argument("--log_path", default="./logs", type=str, help="location to load policy from")
parser.add_argument("--num_episodes", default=10, type=int, help="number of episodes to visualize")
parser.add_argument("--grasp",
                    type=utils.boolify,
                    help="start with grasp",
                    required=True)
parser.add_argument("--safety",
                    type=utils.boolify,
                    help="used safety rewards",
                    required=True)
parser.add_argument("--log_torques", default=False, type=utils.boolify, help="record torques for graphing")

args = parser.parse_args()

task_path = args.log_path+"/"

task_dirs = [f for f in listdir(task_path)]

for task_dir in task_dirs:
    print("_grasp_{grasp}__safety_{safety}".format(grasp=args.grasp, safety=args.safety))
    if "_grasp_{grasp}__safety_{safety}".format(grasp=args.grasp, safety=args.safety) in task_dir:
        true_dir = task_dir


RENDER=True

# set up env options 
controller_config = load_controller_config(default_controller="OSC_POSE")
controller_config["impedance_mode"] = "variable_kp"
controller_config["scale_stiffness"] = True

# set up env options 
options = {}
options["env_name"] = args.task
options["robots"] = "Panda"
options["controller_configs"] = controller_config
options["ee_fixed_to_handle"] = args.grasp

# create and wrap env 
raw_env = suite.make(
    **options,
    has_renderer=RENDER,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping = True,
)



env = BaselinesWrapper(raw_env, safety_rewards = args.safety, save_torques=args.log_torques, grasp_strategy = "max")
env.set_render(RENDER)
obs = env.reset()
eval_returns = 0

initial_qpos = env.robots[0]._joint_positions

object_names=['DoorCIP', 'DoorCIP']
c = 0
success_episodes = 0
unsafe_episodes = 0
for obj in object_names:
    print("Object number")
    print(c)
    c += 1
    POLICYPATH = task_path+ obj +"/model.pt"

    model = TD3("MlpPolicy", env)
    model.policy.load_state_dict(torch.load(POLICYPATH, map_location=torch.device('cpu')))
    model.policy.eval()

# reset the environment
#obs = env.reset()
    eval_returns = 0


    # for i in range(args.num_episodes):
    counter = 0
    stop = False
    while not stop:
        counter += 1
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        eval_returns += reward

        if RENDER:
            env.env.render()
        if (counter % 10 == 0):        #done or 
            stop = True
            if info['unsafe_qpos']:
                unsafe_episodes += 1
            if info['success']:
                success_episodes += 1
            # obs = env.super().reset()
            counter = 0

    print("Done with stuff")


    env.reset() # eventually, don't need this 
    env.render()
    breakpoint()

    # eventually: load new task grasps 
    task = "DoorCIP"
    heuristic_grasps_path = "./grasps/"+task+"_filtered.pkl"
    heuristic_grasps = pickle.load(open(heuristic_grasps_path,"rb"))
    grasp_list, grasp_wp_scores, grasp_qpos_list = list(zip(*heuristic_grasps))
    grasp_idxs = list(range(len(grasp_list)))
    n_grasps = len(grasp_idxs)
    n_qpos_per_grasp = len(grasp_qpos_list[0])

    grasp_list = np.array(grasp_list)                                                   # (n_grasps, 4, 4)
    grasp_qpos_list = np.array(grasp_qpos_list)                                         # (n_grasps, n_qpos_per_grasp, 7)
    grasp_wp_scores = np.array(grasp_wp_scores)                                         # (n_grasps, n_qpos_per_grasp)
    best_wp_scores = np.max(grasp_wp_scores, axis=1)                                    # (n_grasps,)
    best_ik_soln_idx = np.argmax(grasp_wp_scores, axis=1)                               # (n_grasps,)
    best_ik_solns = grasp_qpos_list[np.arange(n_grasps), best_ik_soln_idx]              # (n_grasps, 7)

    best_manipulability = np.argmax(best_wp_scores)
    best_ik_soln = best_ik_solns[best_manipulability]
    env.reset_to_qpos(best_ik_soln)

    # or maybe we need to env.reset_to_grasp rather than env.reset_to_qpos 
    # if we don't have the cache. 
    env.render()




print("Success percentage: {success}".format(success=float(success_episodes)/args.num_episodes))
print("Unsafe percentage: {unsafe}".format(unsafe=float(unsafe_episodes)/args.num_episodes))