import numpy as np
import robosuite as suite
import gym
from gym import spaces
from robosuite.controllers import load_controller_config
import os
import time
from pathlib import Path

import sys
sys.path.append("./baselines_robosuite")
sys.path.append("./rainbow_RBFDQN")

from common import utils, utils_for_q_learning, buffer_class
from common.logging_utils import MetaLogger

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure

from baselines_wrapper import BaselinesWrapper, SaveOnBestTrainingRewardCallback, TrainingSuccessRateCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

import torch
import argparse

RENDER = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name",
                type=str,
                help="Experiment Name",
                required=True)

    parser.add_argument("--run_title",
                type=str,
                help="subdirectory for this run",
                required=True)
    
    parser.add_argument("--bootstrap",
                        type=utils.boolify,
                        help="Should we bootstrap with demonstrations",
                        required=True)

    parser.add_argument("--seed", default=0, help="seed",
                type=int)  # Sets Gym, PyTorch and Numpy seeds

    args, unknown = parser.parse_known_args()

    config = {"training_steps":1000000}

    # create environment instance
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config["impedance_mode"] = "variable_kp" # 7 dim action space
    controller_config["scale_stiffness"] = True

    # set up env options 
    options = {}
    options["env_name"] = "DoorCIP"
    options["robots"] = "Panda"
    options["controller_configs"] = controller_config
    options["ee_fixed_to_handle"] = True

    # create and wrap env 
    raw_env = suite.make(
        **options,
        has_renderer=RENDER,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping = True,
    )

    env = BaselinesWrapper(raw_env)
    env.set_render(RENDER)

    eval_raw_env = suite.make(
        **options,
        has_renderer=RENDER,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping = True,
    )

    eval_env = BaselinesWrapper(eval_raw_env)
    eval_env.set_render(RENDER)

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    directory_to_make = "./baseline_results/"+args.experiment_name + "/" + args.run_title + "_seed_" + str(args.seed)

    Path(directory_to_make).mkdir(parents=True, exist_ok=True)

    log_directory = directory_to_make + "/logs/"
    Path(log_directory).mkdir(parents=True, exist_ok=True)


    logger = configure(directory_to_make, ["stdout", "csv", "log", "tensorboard", "json"])
    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, bootstrap=args.bootstrap)
    
    model.set_random_seed(args.seed)
    model.set_logger(logger)

    # training_success_callback = TrainingSuccessRateCallback(env, logger)
    
    checkpoint_callback = CheckpointCallback(save_freq=100, save_path=log_directory,
                                         name_prefix='checkpoint_')

    # Separate evaluation env
    eval_callback = EvalCallback(eval_env, best_model_save_path=directory_to_make+"/best_model",
                            log_path=directory_to_make, n_eval_episodes=10, eval_freq=10000)
    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])

    model.learn(total_timesteps=config['training_steps'], callback=callback,
    eval_log_path=directory_to_make)