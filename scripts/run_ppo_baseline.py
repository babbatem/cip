import os
import time
import argparse 
import sys 

import numpy as np
import torch
import gym 

import robosuite as suite
from robosuite.controllers import load_controller_config

from stable_baselines3 import PPOBC, PPO
from stable_baselines3.common.monitor import Monitor

from common import utils, utils_for_q_learning
from baselines_robosuite.baselines_wrapper import BaselinesWrapper, SaveOnBestTrainingRewardCallback, RobosuiteEvalCallback, BCCallback

#from stable_baselines3.common.callbacks import EvalCallback

from robo_quickstart import perform_bc

RENDER = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task',   required=True, type=str, 
                        help='name of task. DrawerCIP, DoorCIP, ...')

    parser.add_argument("--seed", default=0, help="seed",
                        type=int)  # Sets Gym, PyTorch and Numpy seeds

    parser.add_argument("--experiment_name",
                        type=str,
                        help="Experiment Name",
                        required=True)

    parser.add_argument("--run_title",
                        type=str,
                        help="subdirectory for this run",
                        required=True)

    parser.add_argument("--grasp",
                        type=utils.boolify,
                        help="start with a grasp?",
                        required=True)
    parser.add_argument("--safety",
                        type=utils.boolify,
                        help="use safety rewards during learning",
                        required=True)
    parser.add_argument("--bc",
                        type=utils.boolify,
                        help="use behavioral cloning?",
                        required=True)

    args = parser.parse_args()
    print(args)

    # TODO: argparse these...
    config = {"training_steps":100000}

    # create logdir
    full_experiment_name = os.path.join(args.experiment_name, args.run_title)
    log_dir = utils.create_log_dir(full_experiment_name)
    
    # create environment instance
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
        reward_shaping=True,
    )
    env = BaselinesWrapper(raw_env, pregrasp_policy=True, safety_rewards=args.safety)
    env = Monitor(env, log_dir, info_keywords=("success","is_success","unsafe_qpos"))
    env.set_render(RENDER)

    #####################
    # create and wrap env 
    eval_raw_env = suite.make(
        **options,
        has_renderer=RENDER,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
    )

    eval_env = BaselinesWrapper(eval_raw_env, pregrasp_policy=True, safety_rewards=args.safety)
    # Use deterministic actions for evaluation
    eval_env = Monitor(eval_env, log_dir+"__evals", info_keywords=("success","is_success","unsafe_qpos"))
    eval_callback = RobosuiteEvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=1000, n_eval_episodes=10,
                                 deterministic=True, render=False)

    #####################

    # seed 
    utils_for_q_learning.set_random_seed({"seed_number" : args.seed, "env" : env})

    all_callbacks = []

    #behavioral cloning?
    if args.bc:
        model = PPOBC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

        """
        pretrain_policy = perform_bc(policy=model.policy, show=False, task=args.task, env=env)
        model.policy = pretrain_policy

        
        behavior_cloning_callback = BCCallback(env, args.task)

        all_callbacks.append(behavior_cloning_callback)
        """
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    all_callbacks.append(eval_callback)

    save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    model.learn(total_timesteps=config["training_steps"],
                callback=all_callbacks)
    #torch.save(model.policy.state_dict(),log_dir+"/model.pt")

