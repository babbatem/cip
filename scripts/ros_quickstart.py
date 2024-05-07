"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

import numpy as np
import pickle

import argparse 
from os import listdir
from os.path import isfile, join

from imitation.data import types


def generater_trajectories(
    policy: rollout.AnyPolicy,
    venv: rollout.VecEnv,
    sample_until: rollout.GenTrajTerminationFn,
    *,
    deterministic_policy: bool = False,
    rng: np.random.RandomState = np.random,
) -> rollout.Sequence[rollout.types.TrajectoryWithRew]:
    """Generate trajectory dictionaries from a policy and an environment.

    Args:
        policy: Can be any of the following:
            1) A stable_baselines3 policy or algorithm trained on the gym environment.
            2) A Callable that takes an ndarray of observations and returns an ndarray
            of corresponding actions.
            3) None, in which case actions will be sampled randomly.
        venv: The vectorized environments to interact with.
        sample_until: A function determining the termination condition.
            It takes a sequence of trajectories, and returns a bool.
            Most users will want to use one of `min_episodes` or `min_timesteps`.
        deterministic_policy: If True, asks policy to deterministically return
            action. Note the trajectories might still be non-deterministic if the
            environment has non-determinism!
        rng: used for shuffling trajectories.

    Returns:
        Sequence of trajectories, satisfying `sample_until`. Additional trajectories
        may be collected to avoid biasing process towards short episodes; the user
        should truncate if required.
    """
    get_actions = rollout._policy_to_callable(policy, venv, deterministic_policy)

    # Collect rollout tuples.
    trajectories = []
    # accumulator for incomplete trajectories
    trajectories_accum = rollout.TrajectoryAccumulator()
    obs = venv.reset()
    for env_idx, ob in enumerate(obs):
        # Seed with first obs only. Inside loop, we'll only add second obs from
        # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
        # get all observations, but they're not duplicated into "next obs" and
        # "previous obs" (this matters for, e.g., Atari, where observations are
        # really big).
        trajectories_accum.add_step(dict(obs=ob), env_idx)

    # Now, we sample until `sample_until(trajectories)` is true.
    # If we just stopped then this would introduce a bias towards shorter episodes,
    # since longer episodes are more likely to still be active, i.e. in the process
    # of being sampled from. To avoid this, we continue sampling until all epsiodes
    # are complete.
    #
    # To start with, all environments are active.
    active = np.ones(venv.num_envs, dtype=bool)
    while np.any(active):
        acts = get_actions(obs)
        obs, rews, dones, infos = venv.step(acts)

        # If an environment is inactive, i.e. the episode completed for that
        # environment after `sample_until(trajectories)` was true, then we do
        # *not* want to add any subsequent trajectories from it. We avoid this
        # by just making it never done.
        dones &= active

        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            acts,
            obs,
            rews,
            dones,
            infos,
        )
        trajectories.extend(new_trajs)

        if sample_until(trajectories):
            # Termination condition has been reached. Mark as inactive any
            # environments where a trajectory was completed this timestep.
            active &= ~dones

    # Note that we just drop partial trajectories. This is not ideal for some
    # algos; e.g. BC can probably benefit from partial trajectories, too.

    # Each trajectory is sampled i.i.d.; however, shorter episodes are added to
    # `trajectories` sooner. Shuffle to avoid bias in order. This is important
    # when callees end up truncating the number of trajectories or transitions.
    # It is also cheap, since we're just shuffling pointers.
    rng.shuffle(trajectories)

    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        exp_obs = (n_steps + 1,) + venv.observation_space.shape
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        exp_act = (n_steps,) + venv.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    return trajectories


def rollouter(
    policy: rollout.AnyPolicy,
    venv: rollout.VecEnv,
    sample_until: rollout.GenTrajTerminationFn,
    *,
    unwrap: bool = True,
    exclude_infos: bool = True,
    verbose: bool = True,
    **kwargs,
) -> rollout.Sequence[rollout.types.TrajectoryWithRew]:
    """Generate policy rollouts.

    The `.infos` field of each Trajectory is set to `None` to save space.

    Args:
        policy: Can be any of the following:
            1) A stable_baselines3 policy or algorithm trained on the gym environment.
            2) A Callable that takes an ndarray of observations and returns an ndarray
            of corresponding actions.
            3) None, in which case actions will be sampled randomly.
        venv: The vectorized environments.
        sample_until: End condition for rollout sampling.
        unwrap: If True, then save original observations and rewards (instead of
            potentially wrapped observations and rewards) by calling
            `unwrap_traj()`.
        exclude_infos: If True, then exclude `infos` from pickle by setting
            this field to None. Excluding `infos` can save a lot of space during
            pickles.
        verbose: If True, then print out rollout stats before saving.
        **kwargs: Passed through to `generate_trajectories`.

    Returns:
        Sequence of trajectories, satisfying `sample_until`. Additional trajectories
        may be collected to avoid biasing process towards short episodes; the user
        should truncate if required.
    """
    trajs = generater_trajectories(policy, venv, sample_until, **kwargs)
    if unwrap:
        trajs = [rollout.unwrap_traj(traj) for traj in trajs]
    if exclude_infos:
        trajs = [rollout.dataclasses.replace(traj, infos=None) for traj in trajs]
    if verbose:
        stats = rollout.rollout_stats(trajs)
        rollout.logging.info(f"Rollout stats: {stats}")
    return trajs


def train_expert():
    print("Training a expert.")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
    )
    expert.learn(10)  # Note: change this to 100000 to trian a decent expert.
    return expert


def sample_expert_transitions():
    expert = train_expert()

    print("Sampling expert transitions.")
    rollouts = rollouter(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=5),
    )

    return rollout.flatten_trajectories(rollouts)

def perform_bc(policy=None, show=False, env=None,twr=None):

    """
    task_path = "./imitation_demos/DoorCIP/"
    task_files = [f for f in listdir(task_path) if isfile(join(task_path, f))]

    expert_demos = []

    for task_file in task_files:
        print(task_path+task_file)
        demo = pickle.load(open(task_path+task_file,"rb"))
        expert_demos.append(demo)
    """
    expert_demos=[twr]

    #transitions = sample_expert_transitions()
    transitions = rollout.flatten_trajectories(expert_demos)

    if policy == None:
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
        )
    else:
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            policy=policy
        )



    if show:
        print("Training a policy using Behavior Cloning")
        bc_trainer.train(n_epochs=2)
        import socket

        HOST = "127.0.0.1"  # The server's hostname or IP address
        PORT = 65444  # The port used by the server
        print("connecting to server")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            while True:
                data = s.recv(1024)
                data=data[1:]
                data=data[:-1]
                data = data.decode().split(",")
                data = [float(x) for x in data]
                print("incoming data")
                print(data)
                action = bc_trainer.policy.predict(data)
                print("action out")
                print(action[0])
                s.sendall(str(action).encode())

        #print(bc_trainer.policy.predict(np.array([0]*27)))
        #print(type(bc_trainer.policy.action_net))

        reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=10, render=True)
        print(f"Reward after training: {reward}")
    else:
        bc_trainer.train(n_epochs=1)
        return(bc_trainer.policy)

if __name__ == "__main__":
    task = "DoorCIP"
    from ros_gym_wrapper import ROSEnv

    env = ROSEnv()

    data = np.load('kuka_door_bc.npy',allow_pickle=True,encoding='bytes')

    obs_ar = np.array(data.item()[b'states'])
    act_ar = np.array(data.item()[b'actions'])
    rews_ar = np.array(data.item()[b'rewards'])
    terminal = True
    infos = None

    twr = types.TrajectoryWithRew(obs=obs_ar, acts=act_ar, rews=rews_ar, terminal=terminal, infos=infos)

    perform_bc(policy=None, show=True, env=env, twr=twr)

