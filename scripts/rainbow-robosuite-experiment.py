"""
Rainbow baseline for CIP. Place in experiments/ folder. 

"""

import argparse
import logging
import os
import datetime
import sys
sys.path.append("..")

from common import utils, utils_for_q_learning, buffer_class
from common.logging_utils import MetaLogger
from rainbow.RBFDQN_rainbow import Net
from rainbow.dis import Net as DistributionalNet
from stochastic_regression import StochasticRegression, load_and_plot_q
import torch
import numpy as np
import numpy
import gym
import robosuite as suite
from gym import spaces
from robosuite.controllers import load_controller_config
import time
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

class Spec(object):
    def __init__(self,id):
        super().__init__()
        self.id = id

class BaselinesWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        #env.action_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(env.robots[0].dof,), dtype=np.float32)
        #env.action_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(7,), dtype=np.float32)
        #variable_kp
        env.action_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(13,), dtype=np.float32)

        #Lift task
        #env.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(42,), dtype=np.float32)
        #Door task
        env.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(46,), dtype=np.float32)
        #DoorCIP task
        #env.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(45,), dtype=np.float32)

        env.reward_range = (-float("inf"), float("inf"))
        env.metadata = {}
        env.spec = Spec("BaselinesWrapper?")
        super().__init__(env)

        self.env = env
        
        self._max_episode_steps = 200

    def set_render(self, render_state):
        self.env.has_renderer = render_state
    def render(self):
        self.env.render()
    def observation(self, obs):
        return(np.concatenate((obs["robot0_proprio-state"],obs["object-state"])))
    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.env.has_renderer:
            self.render()
        return obs, reward, done, info
    def seed(self, seed=None):
        """
        Utility function to set numpy seed
        Args:
            seed (None or int): If specified, numpy seed to set
        Raises:
            TypeError: [Seed must be integer]
        """
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper_param_directory",
                        required=False,
                        default="./hyper_parameters",
                        type=str)

    parser.add_argument("--hyper_parameter_name",
                        required=True,
                        help="0, 10, 20, etc. Corresponds to .hyper file",
                        default="0")  # OpenAI gym environment name

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

    parser.add_argument("--double",
                        type=utils.boolify,
                        default=False,
                        help="run using double DQN")

    parser.add_argument("--nstep",
                        type=int,
                        default=1,
                        help="run using multi-step returns of size n")

    parser.add_argument("--per",
                        type=utils.boolify,
                        default=False,
                        help="run using Priority Experience Replay (PER)")

    parser.add_argument("--dueling",
                        type=utils.boolify,
                        default=False,
                        help="run using dueling architecture")

    parser.add_argument("--mean",
                        type=utils.boolify,
                        default=False,
                        help=
                        """use the mean combine operator to
                        combine advantage and base value
                        (otherwise max is used by default""")

    parser.add_argument("--layer_normalization",
                        type=utils.boolify,
                        default=False,
                        help=
                        """apply normalization immediately
                        prior to activations in any hidden layers""")

    parser.add_argument("--noisy_layers",
                        type=utils.boolify,
                        default=False,
                        help=
                        """use noisy linear layers instead of linear
                        layers for all hidden layers""")

    parser.add_argument("--noisy_where",
                        type=str,
                        default="both",
                        choices=["both","centroid","value"],
                        help=
                        """Specify where noisy layers should be inserted:
                        centroid, value, or both""")

    parser.add_argument("--distributional",
                        type=utils.boolify,
                        default=False,
                        help=
                        """Use Distributional RBF-DQN""")

    parser.add_argument("--alpha", default=0.1, help="alpha",
                        type=float)  # alpha for PER

    parser.add_argument("--per_beta_start", default=0.4, help="beta for per",type=float)  # beta for PER

    parser.add_argument("--should_schedule_beta",
                        type=utils.boolify,
                        default=True,
                        help=
                        """Whether to anneal the value of beta from per_beta_start to 1.0 over the course of training""")
    parser.add_argument("--loss_type",
                        type=str,
                        default="MSELoss",
                        help=
                        """there are two types of loss we can use, MSELoss or HuberLoss""")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=0)
    parser.add_argument("--num_points",
                        type=int,
                        default=0)
    parser.add_argument("--vmin",
                        type=float,
                        default=0)
    parser.add_argument("--vmax",
                        type=float,
                        default=0)
    parser.add_argument("--learning_rate_location_side",
                        type=float,
                        default=0)
    parser.add_argument("--target_network_learning_rate",
                        type=float,
                        default=0)
    parser.add_argument("--regularize_centroid_spread",
                        type=utils.boolify,
                        default=False)
    parser.add_argument("--regularize_centroid_spread_parameter",
                        type=float,
                        default=0)
    parser.add_argument("--regularize_centroid_central_parameter",
                        type=float,
                        default=0)
    parser.add_argument("--regularize_centroid_central",
                        type=utils.boolify,
                        default=False)

    parser.add_argument("--temperature",
                        type=float,
                        required=False)

    parser.add_argument("--log", action="store_true")

    parser.add_argument("--sigma_noise", default=0.5, help="sigma",
                        type=float) ## Noise standard deviation
    parser.add_argument("--policy_type", ## If not specified, reads the policy_type
                        type=str,        ## from the hyperparam file. Otherwise, overrides
                        required=False,  ## with the value from the commandline
                        default="unset")

    parser.add_argument('--which_gpu', '-gpu_id', default=0)

    parser.add_argument("--demo_regression", type=utils.boolify, default=False, required=False)
    parser.add_argument("--load_model", type=str, default="", required=False)

    # use randomly initialized betas for all centroids (fixed) throughout training. 
    parser.add_argument("--random_betas", 
                        type=utils.boolify, 
                        required=False,
                        default=False)

    args, unknown = parser.parse_known_args()
    other_args = {(utils.remove_prefix(key, '--'), val)
                  for (key, val) in zip(unknown[::2], unknown[1::2])}

    full_experiment_name = os.path.join(args.experiment_name, args.run_title)

    utils.create_log_dir(full_experiment_name)
    hyperparams_dir = utils.create_log_dir(
        os.path.join(full_experiment_name, "hyperparams"))
    params = utils_for_q_learning.get_hyper_parameters(args.hyper_parameter_name, "rbf")
    params['hyper_parameters_name'] = args.hyper_parameter_name
    params['full_experiment_file_path'] = os.path.join(os.getcwd(), full_experiment_name)

    for arg_name, arg_value in other_args:
        utils.update_param(params, arg_name, arg_value)

    params['hyperparams_dir'] = hyperparams_dir
    params['start_time'] = str(datetime.datetime.now())
    params['seed_number'] = args.seed
    params['log'] = args.log
    params['per'] = args.per
    params['dueling'] = args.dueling
    params['distributional'] = args.distributional
    params['alpha'] = args.alpha
    params['per_beta_start'] = args.per_beta_start
    params['should_schedule_beta'] = args.should_schedule_beta
    params['loss_type'] = args.loss_type
    params['random_betas'] = args.random_betas
    params['noisy_episode_cutoff'] = params['max_step']/2

    # change hyper parameters from command line
    if args.learning_rate:
        params['learning_rate'] = args.learning_rate
    if args.num_points:
        params['num_points'] = args.num_points
    if args.vmin:
        params['vmin'] = args.vmin
    if args.vmax:
        params['vmax'] = args.vmax
    if args.learning_rate_location_side:
        params['learning_rate_location_side'] = args.learning_rate_location_side
    if args.target_network_learning_rate:
        params['target_network_learning_rate'] = args.target_network_learning_rate
    if args.regularize_centroid_spread_parameter:
        params['regularize_centroid_spread_parameter'] = args.regularize_centroid_spread_parameter

    params['regularize_centroid_spread'] = args.regularize_centroid_spread
    if args.regularize_centroid_central_parameter:
        params['regularize_centroid_central_parameter'] = args.regularize_centroid_central_parameter

    if args.temperature:
        params['temperature'] = args.temperature

    params['regularize_centroid_central'] = args.regularize_centroid_central
    #params['beta'] = args.beta
    params['sigma_noise'] = args.sigma_noise
    if args.policy_type != "unset":
        params['policy_type'] = args.policy_type
    print("Distributional:", params["distributional"])

    # Rainbow RBF-DQN improvements
    params['double'] = args.double
    print("Double:", params["double"])

    if (args.nstep != 1):
        params["nstep"] = True
        params["nstep_size"] = args.nstep
    else:
        params["nstep"] = False
        params["nstep_size"] = -1  # note using multi step returns.

    print("Nstep:", params["nstep"], "Nstep_size:", params["nstep_size"])

    print("PER:", params["per"])
    if params["per"]:
        print("PER alpha:",params['alpha'],"PER beta:",params['per_beta_start'])
        print("Scheduling PER Beta to go from per_beta_start-->1.0:", params['should_schedule_beta'])

    if (args.mean and params['dueling']):
        params["dueling_combine_operator"] = "mean"
    else:
        params["dueling_combine_operator"] = "max"

    print("Dueling:", params['dueling'], "Combine Operator:", params['dueling_combine_operator'])

    params['layer_normalization'] = args.layer_normalization
    params['noisy_layers'] = args.noisy_layers
    if params['noisy_layers']:
        params["layer_normalization"] = True
    params['noisy_where'] = args.noisy_where

    print("Layer Normalizaton: ", params['layer_normalization'], "Noisy Layers: ", params['noisy_layers'])
    print("Noisy Layers Applied to: ", params['noisy_where'])

    utils.save_hyper_parameters(params, args.seed)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.which_gpu))
        print("Using GPU id {}".format(args.which_gpu))
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    #if train is true, render false. If eval, render true
    train = True
    render = False#not train
    #task = "Lift"
    #model_file = "ppo_Lift_300000"
    task = "Door" # DoorCIP uses CIP vs. Door is end to end. 
    rl_algo = "Rainbow_RBF-DQN"
    timestr = time.strftime("%Y%m%d-%H%M%S")

     # create environment instance
    controller_config = load_controller_config(default_controller="OSC_POSE")

    #
    controller_config["impedance_mode"] = "variable_kp"
    controller_config["delta_kp"] = True
    raw_env = suite.make(
        env_name=task, # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping = True,
        controller_configs=controller_config,
    )

    env = BaselinesWrapper(raw_env)
    env.set_render(render)
    
    params['env'] = env

    utils_for_q_learning.set_random_seed(params)
    
    s0 = env.reset()
    utils_for_q_learning.action_checker(env)

    if not params['distributional']:
        Q_object = Net(params,
                       env,
                       state_size=len(s0),
                       action_size=len(env.action_space.low),
                       device=device)
        Q_object_target = Net(params,
                              env,
                              state_size=len(s0),
                              action_size=len(env.action_space.low),
                              device=device)
    else:
        Q_object = DistributionalNet(params,
                                     env,
                                     state_size=len(s0),
                                     action_size=len(env.action_space.low),
                                     device=device)
        Q_object_target = DistributionalNet(params,
                                            env,
                                            state_size=len(s0),
                                            action_size=len(env.action_space.low),
                                            device=device)

    Q_object_target.eval()

    utils_for_q_learning.sync_networks(target=Q_object_target,
                                       online=Q_object,
                                       alpha=params['target_network_learning_rate'],
                                       copy=True)

    if args.demo_regression and args.load_model != "":
        load_and_plot_q(Q_object, args.load_model)
        exit(1)

    print(f"Q_object: {Q_object}")
    print(f"Q_target: {Q_object_target}")

    # Logging with Meta Logger

    meta_logger = MetaLogger(full_experiment_name)
    logging_filename = f"seed_{args.seed}.pkl"

    meta_logger.add_field("evaluation_rewards", logging_filename)
    meta_logger.add_field("episodic_rewards", logging_filename)
    meta_logger.add_field("average_loss", logging_filename)
    meta_logger.add_field("average_q", logging_filename)
    meta_logger.add_field("average_q_star", logging_filename)
    meta_logger.add_field("task_success_rate", logging_filename)

    G_li = []
    loss_li = []
    all_times_per_steps = []
    all_times_per_updates = []

    print("using step based training.")

    # number of steps on average per episode
    env_name_to_steps = {
        "Pendulum-v0": 200, 
        "LunarLanderContinuous-v2":200,
        "BipedalWalker-v3": 1600,
        "Hopper-v3": 1000,
        "HalfCheetah-v3": 1000,
        "Ant-v3": 1000,
        "Humanoid-v2":1000,
        "Walker2d-v2":1000,
        "demo_regression":100,
        "Door":200
    }

    steps_per_typical_episode = env_name_to_steps[params['env_name']]

    steps = 0

    rewards_per_typical_episode = 0

    loss = []
    while (steps <  params['max_step']):

        if (steps%100000 == 0):
            print("step {}".format(steps))

        s, done, t = env.reset(), False, 0
        
        while not done:
            a = Q_object.execute_policy(s, (steps + 1)/steps_per_typical_episode, 'train', steps=(steps+1))
            sp, r, done, _ = env.step(np.nan_to_num(numpy.array(a)))
            
            t = t + 1
            rewards_per_typical_episode += r
            done_p = False if t == env._max_episode_steps else done
            Q_object.buffer_object.append(s, a, r, done_p, sp)
            s = sp

            if t == env._max_episode_steps:
                done = True

            # we do 1 update to the Q network every update_frequency steps. 
            if steps%params['update_frequency'] == 0:
                # now update the Q network
                temp, update_params = Q_object.update(Q_object_target)
                loss.append(temp)
            
            if steps%steps_per_typical_episode == 0:
                # clean up the buffer
                loss_li.append(numpy.mean(loss))
                meta_logger.append_datapoint("average_loss", numpy.mean(loss), write=True)

                meta_logger.append_datapoint("episodic_rewards", rewards_per_typical_episode, write=True)
                rewards_per_typical_episode = 0

                meta_logger.append_datapoint("average_q", update_params['average_q'], write=True)
                meta_logger.append_datapoint("average_q_star", update_params['average_q_star'], write=True)
                loss = []

            if (steps%(10*steps_per_typical_episode) == 0) or (steps == params['max_step'] - 1):
                temp = []

                success_rate = []

                for _ in range(10):
                    s, G, done, t = env.reset(), 0, False, 0
                    while done == False:
                        a = Q_object.execute_policy(s, (steps + 1)/steps_per_typical_episode, 'test', steps=(steps+1))
                        sp, r, done, _ = env.step(np.nan_to_num(numpy.array(a)))
                        s, G, t = sp, G + r, t + 1

                        if t == env._max_episode_steps:
                            done = True
                    temp.append(G)

                    if env.env._check_success():
                        success_rate.append(1)
                    else:
                        success_rate.append(0)
                
                success_rate = np.array(success_rate)

                print(
                    "after {} steps, learned policy collects {} average returns with success rate {}".format(
                        steps, numpy.mean(temp), numpy.mean(success_rate)))

                G_li.append(numpy.mean(temp))
                utils_for_q_learning.save(G_li, loss_li, params, "rbf")
                meta_logger.append_datapoint("evaluation_rewards", numpy.mean(temp), write=True)
                meta_logger.append_datapoint("task_success_rate", numpy.mean(success_rate), write=True)

            if (params["log"] and ((steps % (50*steps_per_typical_episode) == 0) or steps == (params['max_step'] - 1))):
                path = os.path.join(params["full_experiment_file_path"], "logs")
                if not os.path.exists(path):
                    try:
                        os.makedirs(path, exist_ok=True)
                    except OSError:
                        print("Creation of the directory %s failed" % path)
                    else:
                        print("Successfully created the directory %s " % path)
                torch.save(Q_object.state_dict(), os.path.join(path, "step_" + str(steps) + "_seed_" + str(args.seed)))
                torch.save(Q_object_target.state_dict(), os.path.join(path, "target_step_" + str(steps) + "_seed_" + str(args.seed)))

            steps += 1


        # notify n-step that the episode has ended. 
        if (params['nstep']):
            Q_object.buffer_object.storage.on_episode_end()

