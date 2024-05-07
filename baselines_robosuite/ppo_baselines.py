import numpy as np
import robosuite as suite
import gym
from gym import spaces
from robosuite.controllers import load_controller_config
import os
import time
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.monitor import Monitor

from stable_baselines3 import PPO
import torch



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
    def __init__(self, env):
        #env.action_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(env.robots[0].dof,), dtype=np.float32)
        #env.action_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(7,), dtype=np.float32)
        #variable_kp
        env.action_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(13,), dtype=np.float32)

        #Lift task
        #env.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(42,), dtype=np.float32)
        #Door task
        #env.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(46,), dtype=np.float32)
        #Door task + haptics
        env.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(52,), dtype=np.float32)

        env.reward_range = (-float("inf"), float("inf"))
        env.metadata = {}
        env.spec = Spec("BaselinesWrapper?")
        super().__init__(env)

        self.env = env
    def set_render(self, render_state):
        self.env.has_renderer = render_state
    def render(self):
        self.env.render()

    def observation(self, obs):
        return(np.concatenate((obs["robot0_proprio-state"],obs["object-state"])))
    def step(self, action):
        #zero action
        #action = np.zeros(13)
        obs, reward, done, info = super().step(action)
        if self.env.has_renderer:
            self.render()
        return obs, reward, done, info


def train_robosuite(config):
    #if train is true, render false. If eval, render true
    train = True
    render = not train
    save = True
    #task = "Lift"
    #model_file = "ppo_Lift_300000"
    task = "Door"#"DoorCIP"
    rl_algo = "PPO"
    timestr = time.strftime("%Y%m%d-%H%M%S")

    log_path = "./logs"
    if train:
        log_name = task+"_"+rl_algo+"_"+timestr
    else:
        #log_name = "Door_PPO_20220316-113318" #good for door task no impedance action space or haptics
        log_name = "Door_PPO_20220329-132725"
    log_dir = log_path+"/"+log_name



    if train and save:
        os.mkdir(log_dir)




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

    #raw_env.ee_fixed_to_handle = False


    env = BaselinesWrapper(raw_env)
    env = Monitor(env, log_dir)
    env.set_render(render)
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)


    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)


    if train:
        model.learn(total_timesteps=config["training_steps"],callback=callback)
            #eval_log_path=log_path+log_dir)
        #save model
        torch.save(model.policy.state_dict(),log_dir+"/model.pt")
    else:
        model.policy.load_state_dict(torch.load(log_dir+"/model.pt", map_location=torch.device('cpu')))
        model.policy.eval()


    # reset the environment
    obs = env.reset()
    eval_returns = 0
    for i in range(10000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        eval_returns += reward
        if render:
            env.env.render()
        if done:
            obs = env.reset()

if __name__ == "__main__":
    config = {"training_steps":300000}
    train_robosuite(config)
