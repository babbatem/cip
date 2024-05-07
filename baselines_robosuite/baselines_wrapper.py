import os 
import sys 
import numpy as np 
import torch
import gym
import matplotlib
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy

from gym import spaces 

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EventCallback
from typing import Any, Callable, Dict, List, Optional, Union
# from robo_quickstart import perform_bc
import math


import random

class BCCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, env, task, retrain_freq: int = 500,verbose=0):
        self.retrain_freq = retrain_freq
        self.env = env
        self.task = task
        super(BCCallback, self).__init__(verbose)


    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.n_calls % self.retrain_freq == 0:
            bc_policy = perform_bc(policy=self.model.policy, show=False, task=self.task, env=self.env)
            self.model.policy = bc_policy
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


class RobosuiteEvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    :param max_eval_episodes: terminate learning after this number of eval_episodes
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        max_eval_episodes:int = 500,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.best_success_rate = -np.inf
        self.last_success_rate = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self._is_violation_buffer = []
        self.evaluations_successes = []
        self.evaluations_violations= []
        self.last_eval_episode = 0
        self.eval_episode_frequency = 10
        self.max_eval_episodes = max_eval_episodes


    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]


        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)
            maybe_is_violation = info.get("unsafe_qpos")
            if maybe_is_violation is not None:
                self._is_violation_buffer.append(maybe_is_violation)
            

    def check_eval_freq(self):
        with open(self.best_model_save_path+"/monitor.csv", 'r') as fp:
            num_lines = len(fp.readlines())
            num_training_episodes = num_lines - 2 #First two are header information
            #If we are at the right number of training episodes for eval frequency AND we haven't do eval yet for this episode
            if (num_training_episodes % self.eval_episode_frequency == 0) and (self.last_eval_episode != num_training_episodes):
                print("Doing eval, num training episodes is {} and last eval was at {}!".format(num_training_episodes,self.last_eval_episode))
                self.last_eval_episode = num_training_episodes
                return(True)
            else:
                return(False)
    def _on_step(self) -> bool:
        continue_training = True

        if self.check_eval_freq():
        #if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            # Reset success rate buffer
            self._is_success_buffer = []
            self._is_violation_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    self.evaluations_violations.append(self._is_violation_buffer)
                    kwargs = dict(successes=self.evaluations_successes, safety_violations=self.evaluations_violations)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                self.last_success_rate = success_rate
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            if len(self._is_violation_buffer) > 0:
                violation_rate = np.mean(self._is_violation_buffer)
                if self.verbose > 0:
                    print(f"Violation rate: {100 * violation_rate:.2f}%")
                self.logger.record("eval/violation_rate", violation_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                self.best_mean_reward = mean_reward

            try:
                if success_rate > self.best_success_rate:
                    if self.verbose > 0:
                        print("New best mean success rate!")
                    if self.best_model_save_path is not None:
                        #self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                        torch.save(self.model.policy.state_dict(),self.best_model_save_path+"/model.pt")
                    self.best_success_rate = success_rate
                    # Trigger callback on new best model, if needed
                    if self.callback_on_new_best is not None:
                        continue_training = self.callback_on_new_best.on_step()
            except:
                pass

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

            # maybe terminate based on num episodes
            if self.last_eval_episode >= self.max_eval_episodes:
                print(f"reached {self.last_eval_episode} episodes, terminating")
                continue_training = False


        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


class TrainingSuccessRateCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, env, logger, verbose=0):
        super(TrainingSuccessRateCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.env = env
        self.logger = logger

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        
        success = self.env.env._check_success()
        self.logger.record("training_success_rate_episode_end", int(success))


    
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
        


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = log_dir
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  #self.model.save(self.save_path)
                  torch.save(self.model.policy.state_dict(),self.save_path+"/model.pt")

        return True


class Spec(object):
    def __init__(self,id):
        super().__init__()
        self.id = id

class BaselinesWrapper(gym.ObservationWrapper):
    def __init__(self,
                 env, 
                 safety_rewards = True, 
                 safety_violation_penalty=0, 
                 pregrasp_policy = True, 
                 save_torques=False,
                 grasp_strategy=None,
                 ik_strategy="random", # "max", "weighted"
                 use_cached_qpos=False,
                 learning=True,
                 terminate_when_lost_contact=False,
                 num_steps_lost_contact=500,
                 optimal_ik=False,
                 control_gripper=False,
                 task_success_rate_path=None,
                 confidence=1):

        # setup action space 
        low_limits = env.action_spec[0]
        high_limits = env.action_spec[1]
        a_space_shape = env.action_spec[0].shape
        env.action_space = spaces.Box(low=low_limits, high=high_limits, shape=a_space_shape, dtype=np.float32)

        #penalty for safety violation
        self.safety_violation_penalty = safety_violation_penalty
        self.safety_rewards = safety_rewards
        self.pregrasp_policy = pregrasp_policy
        self.save_torques=save_torques
        self.num_broken = 0
        self.num_steps_lost_contact = num_steps_lost_contact

        #max absolute torque, penalized to prevent unsafe behavior
        self.torque_limit = 60

        #for tracking torques 
        self.torque_history = []

        # compute obs shape
        obs = self.observation(env.reset())
        obs_shape = obs.shape

        env.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=obs_shape, dtype=np.float32)

        env.reward_range = (-float("inf"), float("inf"))
        env.metadata = {}
        env.spec = Spec("BaselinesWrapper?")
        super().__init__(env)

        self.env = env
        self._max_episode_steps = 250
        self.grasp_strategy=grasp_strategy
        self.learning = learning 
        self.use_cached_qpos = use_cached_qpos 
        self.optimal_ik = optimal_ik
        self.ik_strategy = ik_strategy
        self.grasp_list = None
        self.terminate_when_lost_contact = terminate_when_lost_contact
        self.control_gripper = control_gripper
        self.task_success_rate_path = task_success_rate_path
        self.confidence = confidence
        assert not (self.terminate_when_lost_contact and not self.env.ee_fixed_to_handle), \
            """
                Learning from scratch and terminating when lost contact not allowed. 
                Either self env.ee_fixed_to_handle True or terminate_when_lost_contact to False. 
            """

        assert ik_strategy in ("random", "max")
        assert grasp_strategy in (None, "tsr_ucb", "fixed") # TODO: support for max grasp strategy w/ object randomization.  
        assert use_cached_qpos == False 

        if self.ik_strategy == "max":
            self.optimal_ik = True 

    def load_tsr_dict(self):
        #the tsr_grasp_dict keys are grasp_idx, values are a list [failure_count,success_count]
        try: #check if there is a task success pickle file already
            tsr_grasp_dict = pickle.load(open(self.task_success_rate_path,"rb"))
        except: #there is not, we build an empty one
            tsr_grasp_dict = {}
            for grasp_idx in self.grasp_idxs: 
                tsr_grasp_dict[grasp_idx] = [0,0] #every grasp starts with 0 failures/success
        return(tsr_grasp_dict)
        
    def reset(self):

        self.pre_grasp_complete = False
        self.contact_hist = [True]*self.num_steps_lost_contact

        if self.env.ee_fixed_to_handle:

            # load grasps 
            if self.learning:
                if self.grasp_list is None:
                    if "Visualization" in self.env.__class__.__name__: #you're collecting demos with visualization wrapper, go one deeper to get true class
                        task = self.env.env.__class__.__name__
                    else:
                        task = self.env.__class__.__name__
                    heuristic_grasps_path = "./grasps/"+task+"_filtered.pkl"
                    heuristic_grasps = pickle.load(open(heuristic_grasps_path,"rb"))
                    grasp_list, self.grasp_wp_scores, self.grasp_qpos_list = list(zip(*heuristic_grasps))
                    self.grasp_idxs = list(range(len(grasp_list)))
                    self.n_grasps = len(self.grasp_idxs)
                    self.n_qpos_per_grasp = len(self.grasp_qpos_list[0])

                    # TODO: solve IK after each reset to compute current wp scores. 

                    grasp_list = np.array(grasp_list)                                                        # (self.n_grasps, 4, 4)
                    self.grasp_qpos_list = np.array(self.grasp_qpos_list)                                    # (self.n_grasps, self.n_qpos_per_grasp, 7)
                    self.grasp_wp_scores = np.array(self.grasp_wp_scores)                                    # (self.n_grasps, self.n_qpos_per_grasp)
                    best_wp_scores = np.max(self.grasp_wp_scores, axis=1)                                    # (self.n_grasps,)
                    best_ik_soln_idx = np.argmax(self.grasp_wp_scores, axis=1)                               # (self.n_grasps,)
                    best_ik_solns = self.grasp_qpos_list[np.arange(self.n_grasps), best_ik_soln_idx]         # (self.n_grasps, 7)

                    self.og_grasp_qpos_list = deepcopy(self.grasp_qpos_list)
                    self.og_grasp_wp_scores = deepcopy(self.grasp_wp_scores)

                    if self.ik_strategy == "max":
                        self.grasp_wp_scores = best_wp_scores
                        self.grasp_qpos_list = best_ik_solns

                    self.grasp_list = grasp_list

            else:
                assert self.grasp_strategy == "fixed"

            if self.grasp_strategy == "tsr_ucb":
                tsr_grasp_dict = self.load_tsr_dict()

            keep_resetting = True
            while keep_resetting:

                # maybe shuffle IK solutions for each grasp. 
                if self.ik_strategy == "random" and self.learning:

                    random_ik_soln_idx = np.random.randint(0, self.n_qpos_per_grasp, size=self.n_grasps)
                    random_ik_solns = self.og_grasp_qpos_list[np.arange(self.n_grasps), random_ik_soln_idx]
                    self.grasp_wp_scores = self.og_grasp_wp_scores[np.arange(self.n_grasps), random_ik_soln_idx]
                    self.grasp_qpos_list = random_ik_solns
                
                #Set grasp_pose for environment, assumed to be world frame. 
                if self.grasp_strategy == None: 
                    sampled_pose_idx = random.choice(self.grasp_idxs)
                    sampled_pose = self.grasp_list[sampled_pose_idx]
                elif self.grasp_strategy == "weighted":
                    sampled_pose_idx = random.choices(self.grasp_idxs, weights=self.grasp_wp_scores, k=1)[0]
                    sampled_pose = self.grasp_list[sampled_pose_idx]
                elif self.grasp_strategy == "tsr_ucb":
                    tsr_scores = [] #task success rate for each grasp, represents Q values
                    n_t = [] #number of times we have tried this grasp
                    t = 1 #total number of times we've tried to pull grasps
                    for grasp_idx in self.grasp_idxs:
                        #calculate tsr_scores
                        if tsr_grasp_dict[grasp_idx][0] == tsr_grasp_dict[grasp_idx][1] and tsr_grasp_dict[grasp_idx][0] == 0: #all values are 0, tsr is 0
                            tsr_scores.append(0)
                        else:
                            tsr = float(tsr_grasp_dict[grasp_idx][1]) / (tsr_grasp_dict[grasp_idx][0] + tsr_grasp_dict[grasp_idx][1])
                            tsr_scores.append(tsr)
                        #calculate n_t
                        n_t.append(tsr_grasp_dict[grasp_idx][0] + tsr_grasp_dict[grasp_idx][1] + 1) #NOTE 1 is added here because otherwise we get divide by 0 in ucb calculation
                        t += tsr_grasp_dict[grasp_idx][0]
                        t += tsr_grasp_dict[grasp_idx][1] 

                    ucb_values = []
                    for grasp_idx in self.grasp_idxs:
                        ucb_value = tsr_scores[grasp_idx] + self.confidence * np.sqrt(math.log(t) / float(n_t[grasp_idx]))
                        ucb_values.append(ucb_value)

                    sampled_pose_idx = np.argmax(ucb_values)
                    sampled_pose = self.grasp_list[sampled_pose_idx]
                    self.cur_sampled_pose_idx = sampled_pose_idx
                elif self.grasp_strategy == "max":
                    sampled_pose_idx = np.argmax(self.grasp_wp_scores)
                    sampled_pose = self.grasp_list[sampled_pose_idx]
                elif self.grasp_strategy == "fixed":#just use the saved grasp pose, mostly for the filter_cip_grasps script
                    sampled_pose = self.grasp_pose

                o = super().reset()

                if self.use_cached_qpos:

                    # TODO: deprecate? recompute?  
                    sampled_qpos = self.grasp_qpos_list[sampled_pose_idx]
                    self.grasp_success = self.reset_to_qpos(sampled_qpos, wide=True)
                    if self.grasp_success:
                        keep_resetting = False
                else:
                    self.grasp_success = self.reset_to_grasp(
                                            sampled_pose, wide=True, 
                                            optimal_ik=self.optimal_ik,
                                            frame=self.get_obj_pose(),
                                            verbose=self.env.has_renderer
                                         )
                    if self.grasp_success:   
                        keep_resetting = False

                if not self.learning:
                    keep_resetting = False
        else:
            o = super().reset()
            self.sim.forward()

        if len(self.torque_history) > 0 and self.save_torques :
            pickle.dump(np.array(self.torque_history).T, open('torque_data.pkl', 'wb'))
            print("Torques recorded in torque_data.pkl")
            exit(0)

        if self.pregrasp_policy:
            o = self.execute_pregrasp()
        return(o)

    def execute_pregrasp(self):
        # close gripper for a few frames
        a = np.zeros(self.env.action_spec[0].shape)
        a[-1] = 1
        for _ in range(10):
            o, r, d, i = self.step(a)

        self.pre_grasp_complete = True
        return o 

    def set_render(self, render_state):
        self.env.has_renderer = render_state
    def render(self, mode=None):
        self.env.render()
    def observation(self, obs):
        return(np.concatenate((obs["robot0_proprio-state"],obs["object-state"])))
    def step(self, action):
        if not self.control_gripper:
            action[-1] = 1
        obs, reward, done, info = super().step(action)
        if self.env.has_renderer:
            self.render()

        # check arm qpos
        # TODO: tolerance is hardcoded 0.1 rad
        self.torque_history.append(self.env.robots[0].torques)
        lost_contact = False 
        if self.pre_grasp_complete and self.terminate_when_lost_contact:

            self.contact_hist.append(self.check_gripper_contact(self.env.__class__.__name__))
            last_five_contacts = self.contact_hist[-self.num_steps_lost_contact:]
            lost_contact = not np.any(last_five_contacts)
            if lost_contact:
                done = True
                if self.grasp_strategy == "tsr_ucb":
                    tsr_grasp_dict = self.load_tsr_dict()
                    tsr_grasp_dict[self.cur_sampled_pose_idx][0] += 1
                    with open(self.task_success_rate_path, "wb") as pkl_file:
                        pickle.dump(tsr_grasp_dict, pkl_file)    
                print("=============================LOSTCONTACT=================================") 
            info["lost_contact"] = lost_contact


        if self.env.robots[0].check_q_limits():
            done = True
            info["unsafe_qpos"] = True
            self.num_broken += 1
            if self.grasp_strategy == "tsr_ucb":
                tsr_grasp_dict = self.load_tsr_dict()
                tsr_grasp_dict[self.cur_sampled_pose_idx][0] += 1
                with open(self.task_success_rate_path, "wb") as pkl_file:
                    pickle.dump(tsr_grasp_dict, pkl_file)
            print("=============================BROKENROBOT=================================")
            print(self.num_broken)
            # self.logger.record("was_joint_safety_violated", 1)
            #Penalize unsafe qpos
            # if self.safety_rewards:
            #     reward += self.safety_violation_penalty
        else:
            info["unsafe_qpos"] = False
            # self.logger.record("was_joint_safety_violated", 0)
        """
        if np.any(np.abs(self.env.robots[0].torques) > self.torque_limit):
            print("Violated!")
            done = True
            info["unsafe_torque"] = True

            if self.safety_rewards:
                reward += self.safety_violation_penalty
        else:
            info["unsafe_qpos"] = False
        """
        info["is_success"] = self.env._check_success()
        if info["is_success"]:
            done = True
            if self.grasp_strategy == "tsr_ucb":
                tsr_grasp_dict = self.load_tsr_dict()
                tsr_grasp_dict[self.cur_sampled_pose_idx][1] += 1
                with open(self.task_success_rate_path, "wb") as pkl_file:
                    pickle.dump(tsr_grasp_dict, pkl_file)

        # TODO: check gripper with self.env.robots[0].gripper._joints, self.env.sim.model.jnt_range

        # case in which done is True due to horizon, update grasp as failure 
        if done and not info["is_success"] and not self.env.robots[0].check_q_limits() and not lost_contact:
            print('horizon reached, updating ucb')
            if self.grasp_strategy == "tsr_ucb":
                tsr_grasp_dict = self.load_tsr_dict()
                tsr_grasp_dict[self.cur_sampled_pose_idx][0] += 1
                with open(self.task_success_rate_path, "wb") as pkl_file:
                    pickle.dump(tsr_grasp_dict, pkl_file)

        
        return obs, reward, done, info
            

    def seed(self, seed=None):
        """ set numpy seed etc. directly instead. """
        pass
