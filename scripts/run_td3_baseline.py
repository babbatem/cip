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

from stable_baselines3 import PPO, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

from common import utils, utils_for_q_learning
from baselines_robosuite.baselines_wrapper import BaselinesWrapper, SaveOnBestTrainingRewardCallback, RobosuiteEvalCallback

#from stable_baselines3.common.callbacks import EvalCallback 

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
    parser.add_argument("--lr",
                        type=float,
                        help="learning rate",
                        default=0.001)
    parser.add_argument("--noise_std",
                        type=float,
                        help="std for action noise",
                        default=0.1)
    parser.add_argument("--batch_size",
                        type=int,
                        help="batch_size",
                        default=256)
    parser.add_argument("--gradient_steps",
                        type=int,
                        help="number of gradient steps to do (-1 means based on number of steps in rollout)",
                        default=-1)
    parser.add_argument("--grasp_strategy",
                        type=str,
                        help="choose from (max, weighted)",
                        default="weighted")
    parser.add_argument("--num_contact_steps",
                        type=int,
                        help="Terminate after...",
                        default=50)
    parser.add_argument("--action_scale",
                        type=float,
                        help="scale max possible delta action",
                        default=1.0)
    parser.add_argument("--ik_strategy",
                        type=str,
                        help="max or random",
                        default="max")
    parser.add_argument("--render",
                        type=utils.boolify,
                        help="Terminate after...",
                        default=False)
    parser.add_argument("--control_gripper",
                        type=utils.boolify,
                        help="Control the gripper, or instead always have it close",
                        default=True)
    parser.add_argument("--n_eval_episodes",
                        type=int,
                        help="Number of evaluation episodes per eval",
                        default=10)
    parser.add_argument("--max_eval_eps",
                        type=int,
                        help="Number of evaluation episodes per eval",
                        default=500)
    parser.add_argument("--max_num_demos",
                        type=int,
                        help="Number of demos to use",
                        default=100)
    parser.add_argument("--demo_path",
                        type=str,
                        default='auto_demos/')
    args = parser.parse_args()
    print(args)

    # TODO: argparse these...
    config = {"training_steps":1e9}

    # create logdir
    full_experiment_name = os.path.join(args.experiment_name, args.run_title)
    log_dir = utils.create_log_dir(full_experiment_name)

    # log args
    run_dict = vars(args)
    run_dict_file = os.path.join(log_dir, "config.pkl")
    tsr_dict_file = os.path.join(log_dir, "task_success_rates.pkl")
    try:
        os.remove(tsr_dict_file)
    except:
        pass

    with open(run_dict_file, "wb") as pkl_file:
        pickle.dump(run_dict, pkl_file)
    
    # create environment instance
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config["impedance_mode"] = "variable_kp"
    controller_config["scale_stiffness"] = True
    controller_config["safety_bool"] = args.safety
    controller_config["action_scale_param"] = args.action_scale


    # set up env options 
    options = {}
    options["env_name"] = args.task
    options["robots"] = "Panda"
    options["controller_configs"] = controller_config
    options["ee_fixed_to_handle"] = args.grasp
    options["hard_reset"] = True if not args.render else False
    use_cached_qpos = False

    if args.grasp_strategy == "None": args.grasp_strategy = None 
    assert args.grasp_strategy in ["weighted", "max", "tsr_ucb", None]

    # create and wrap env 
    raw_env = suite.make(
        **options,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
    )
    env = BaselinesWrapper(raw_env, 
                           pregrasp_policy=args.grasp, 
                           safety_rewards=args.safety,
                           grasp_strategy=args.grasp_strategy,
                           use_cached_qpos=use_cached_qpos, 
                           terminate_when_lost_contact=args.grasp,
                           num_steps_lost_contact=args.num_contact_steps,
                           ik_strategy=args.ik_strategy,
                           control_gripper=args.control_gripper,
                           task_success_rate_path=tsr_dict_file)
    env = Monitor(env, log_dir, info_keywords=("success","is_success","unsafe_qpos"))
    env.set_render(args.render)

    #####################
    # create and wrap env 
    eval_raw_env = suite.make(
        **options,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
    )

    eval_env = BaselinesWrapper(eval_raw_env, 
                                pregrasp_policy=args.grasp, 
                                safety_rewards=args.safety,
                                grasp_strategy=args.grasp_strategy,
                                use_cached_qpos=use_cached_qpos, 
                                terminate_when_lost_contact=args.grasp,
                                num_steps_lost_contact=args.num_contact_steps,
                                ik_strategy=args.ik_strategy,
                                control_gripper=args.control_gripper,
                                task_success_rate_path=tsr_dict_file)

    # Use deterministic actions for evaluation
    eval_callback = RobosuiteEvalCallback(eval_env, 
                                          best_model_save_path=log_dir,
                                          log_path=log_dir, 
                                          eval_freq=1000, 
                                          n_eval_episodes=args.n_eval_episodes,
                                          deterministic=True, 
                                          render=False,
                                          max_eval_episodes=args.max_eval_eps)

    #####################

    # seed 
    utils_for_q_learning.set_random_seed({"seed_number" : args.seed, "env" : env})

    all_callbacks = []

    n_actions = eval_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.noise_std * np.ones(n_actions))
    demo_path = f"./{args.demo_path}/{args.task}"
    model = TD3("MlpPolicy", env, 
                action_noise=action_noise, 
                verbose=1, 
                bootstrap=args.bc, 
                tensorboard_log=log_dir,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                gradient_steps=args.gradient_steps,
                seed=args.seed,
                demo_path=demo_path,
                max_num_demos=args.max_num_demos)
   
    all_callbacks.append(eval_callback)

    save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    # maybe BC pretraining 
    # model.train_bc(1000)
    # breakpoint()
    # evaluate_policy(model, env)
    # breakpoint()

    model.learn(total_timesteps=config["training_steps"],
                callback=all_callbacks)
    print(f'training complete. num timesteps: {model.num_timesteps}')
    #torch.save(model.policy.state_dict(),log_dir+"/model.pt")

