import gym
from gym import spaces
import numpy as np

import socket

class ROSEnv(gym.Env):
  def __init__(self, ros_tcp_ip="127.0.0.1"):
    super(ROSEnv, self).__init__()

    self.HOST = ros_tcp_ip  # The server's hostname or IP address
    self.PORT = 65432  # The port used by the server TODO make sure host has same port

    low_limits = -0.0001 #TODO: Low limit
    high_limits = 0.0001 #TODO: High limit
    a_space_shape = (3,) #TODO: shape of action space
    obs_shape = (27,)

    self.action_space = spaces.Box(low=low_limits, high=high_limits, shape=a_space_shape, dtype=np.float32)

    #TODO make sure this makes sense
    self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=obs_shape, dtype=np.float32)


  def step(self, action):
    # Execute one time step within the environment
    #TODO take a step, return state, 
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((self.HOST, self.PORT))
        s.sendall(str(action))
        data = s.recv(1024)
        #TODO parse data into obs, reward, done, info

    obs = 0 #TODO
    reward = 0 #TODO
    done = 0 #TODO
    info = 0 #TODO
    return(obs, reward, done, info)
  def reset(self):
    # Reset the state of the environment to an initial state
    #TODO reset and pass back initial state
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((self.HOST, self.PORT))
        s.sendall("reset")
        data = s.recv(1024)
        #TODO parse data into obs
    obs = 0
    return(obs)
  """
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    pass
  """