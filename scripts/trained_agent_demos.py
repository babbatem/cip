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

def collect_demos(env, model, args, horizon=250):
    """
    Collect demonstrations using the model provided. 

    TODO: guarantee success. 
    """
    success_episodes = 0
    unsafe_episodes = 0
    episode_counter = 0 

    while episode_counter < args.num_episodes:

        expert_buffer = []
        counter = 0
        stop = False

        s0 = env.reset()

        while not stop:
            counter += 1
            action, _state = model.predict(s0, deterministic=True)
            sp, reward, done, info = env.step(action)
            expert_buffer.append((s0, action, reward, done, sp))
            s0 = sp

            if args.render:
                env.env.render()
            if done or (counter % horizon == 0):
                stop = True
                if info['unsafe_qpos']:
                    unsafe_episodes += 1
                if info['success']:
                    success_episodes += 1

        # maybe guarantee success 
        if args.guarantee:
            if not info['success']:
                print('got failure')
                continue
        print('got success')

        # save the demo 
        t1, t2 = str(time.time()).split(".")
        filepath = "./e2e_auto_demos/{}/".format(args.task)+"{}_{}".format(t1, t2)+"_"+str(episode_counter)+".p"
        pickle.dump(expert_buffer, open(filepath,"wb"))

        episode_counter +=1

    print("Successes {success}".format(success=success_episodes))
    print("Unsafe episodes {unsafe}".format(unsafe=unsafe_episodes))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task',   required=True, type=str, 
                        help='name of task. DrawerCIP, DoorCIP, ...')
    parser.add_argument("--log_path", default="./logs", type=str, help="location to load policy from")
    parser.add_argument("--num_episodes", default=10, type=int, help="number of episodes to visualize")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--grasp", action="store_true")
    parser.add_argument("--guarantee", action="store_true", help="guarantee args.num_episodes successes")

    args = parser.parse_args()

    task_path = args.log_path+"/"
    policypath = task_path +"/model.pt"

    # set up env options 
    # TODO: these should load from job config, esp. action scale...
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config["impedance_mode"] = "variable_kp"
    controller_config["scale_stiffness"] = True
    controller_config["safety_bool"] = False
    controller_config["action_scale_param"] = 0.5

    # set up env options 
    options = {}
    options["env_name"] = args.task
    options["robots"] = "Panda"
    options["controller_configs"] = controller_config
    options["ee_fixed_to_handle"] = args.grasp

    # create and wrap env 
    raw_env = suite.make(
        **options,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping = True,
    )
    env = BaselinesWrapper(raw_env, 
                           pregrasp_policy=False, 
                           safety_rewards=False,
                           grasp_strategy=None,
                           use_cached_qpos=False, 
                           terminate_when_lost_contact=False,
                           num_steps_lost_contact=int(1e3),
                           ik_strategy="max",
                           control_gripper=True)
    env.set_render(args.render)

    model = TD3("MlpPolicy", env)
    model.policy.load_state_dict(torch.load(policypath, map_location=torch.device('cpu')))
    model.policy.eval()

    collect_demos(env, model, args)