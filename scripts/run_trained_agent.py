import os
import time
import argparse 
import sys 

import numpy as np
import torch
import gym 

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
POLICYPATH = task_path +"/model.pt"

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
env = BaselinesWrapper(raw_env, safety_rewards = args.safety, save_torques=args.log_torques, pregrasp_policy=args.grasp)
env.set_render(RENDER)

model = TD3("MlpPolicy", env)
model.policy.load_state_dict(torch.load(POLICYPATH, map_location=torch.device('cpu')))
model.policy.eval()

# reset the environment
obs = env.reset()
eval_returns = 0

success_episodes = 0
unsafe_episodes = 0
for i in range(args.num_episodes):
    counter = 0
    stop = False
    while not stop:
        counter += 1
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        eval_returns += reward

        if RENDER:
            env.env.render()
        if done or (counter % 250 == 0):
            stop = True
            if info['unsafe_qpos']: 
                unsafe_episodes += 1
            if info['success']:
                success_episodes += 1
            obs = env.reset()
            counter = 0

print("Success percentage: {success}".format(success=float(success_episodes)/args.num_episodes))
print("Unsafe percentage: {unsafe}".format(unsafe=float(unsafe_episodes)/args.num_episodes))