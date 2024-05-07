import os
import time
import argparse 
import sys 
import pickle 
import copy

import numpy as np
import torch
import gym 
from tqdm import tqdm

import robosuite as suite
from robosuite.controllers import load_controller_config

from stable_baselines3 import TD3
from baselines_robosuite.baselines_wrapper import BaselinesWrapper

def main(args):

    t0 = time.time()

    RENDER = False
    N_Q_PER_GRASP = 100
    NUM_ROLLOUTS = 10
    POLICYPATH=args.policy
    OUTDIR = os.path.dirname(args.policy)

    config_path = os.path.join(OUTDIR, 'config.pkl')                
    with open(config_path, 'rb') as f:
        job_data = pickle.load(f)
    
    # set up env options 
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config["impedance_mode"] = "variable_kp"
    controller_config["scale_stiffness"] = True
    controller_config["safety_bool"] = False
    controller_config["action_scale_param"] = job_data['action_scale']

    # set up env options 
    options = {}
    options["env_name"] = args.task
    options["robots"] = "Panda"
    options["controller_configs"] = controller_config
    options["ee_fixed_to_handle"] = True
    options["hard_reset"] = False

    # create and wrap env 
    raw_env = suite.make(
        **options,
        has_renderer=RENDER,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping = True,
    )
    env = BaselinesWrapper(raw_env, 
                           pregrasp_policy=False, 
                           safety_rewards=False,
                           grasp_strategy="fixed",
                           optimal_ik=False, 
                           use_cached_qpos=False, 
                           learning=False, 
                           terminate_when_lost_contact=True,
                           num_steps_lost_contact=50,
                           ik_strategy="random",
                           control_gripper=job_data['control_gripper'])
    env.set_render(RENDER)

    model = TD3("MlpPolicy", env)
    model.policy.load_state_dict(torch.load(POLICYPATH, map_location=torch.device('cpu')))
    model.policy.eval()

    # load grasps, solve IK with different seeds 
    heuristic_grasps_path = "./grasps/"+args.task+"_filtered.pkl"
    heuristic_grasps = pickle.load(open(heuristic_grasps_path,"rb"))
    grasp_list, grasp_wp_scores, grasp_qpos_list = list(zip(*heuristic_grasps))
    
    data = []
    metadata=[]
    for (grasp_idx, grasp) in enumerate(grasp_list):
        print('\n\n GRASP {} \n\n'.format(grasp_idx))
        i=0
        # for i in range(N_Q_PER_GRASP):
        while i < N_Q_PER_GRASP:

            print(f' Q ATTEMPT {i}')
            
            env.grasp_pose = grasp
            obs = env.reset()
            test_qpos = copy.deepcopy(env.sim.data.qpos[:7])
            w,p,wp = env.check_manipulability()
            if not env.grasp_success:
                print('GRASP FAILED')
                continue

            i+=1
            obs = env.execute_pregrasp()

            # run and evaluate success 
            eval_returns = 0
            success_episodes = 0
            unsafe_episodes = 0
            for _ in range(NUM_ROLLOUTS):

                states  = []
                actions = []
                counter = 0
                stop = False
                while not stop:
                    counter += 1

                    states.append(obs)
                    action, _state = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    eval_returns += reward
                    actions.append(action)

                    if RENDER:
                        env.env.render()
                    if done or (counter % 250 == 0):
                        stop = True
                        if info['unsafe_qpos']:
                            unsafe_episodes += 1
                        if info['success']:
                            success_episodes += 1

                        states.append(obs)
                        datum = (grasp, test_qpos, w, p, wp, states, actions, info['success']) 
                        data.append(datum)

                        obs = env.reset()
                        is_valid = env.reset_to_qpos(test_qpos, wide=True)
                        obs = env.execute_pregrasp()
                        counter = 0


            success = float(success_episodes) / NUM_ROLLOUTS
            unsafe = float(unsafe_episodes) / NUM_ROLLOUTS
            print("\tQ ATTEMPT {}".format(i))
            print("\t\tSuccess percentage: {}".format(success))
            print("\t\tUnsafe percentage: {}".format(unsafe))
            metadata.append((grasp, success, unsafe))

    t1 = time.time()
    print('Total time: {} min'.format( (t1-t0)/60.) )
    print('Time per grasp: {} min'.format( (t1-t0) / 60. / len(data)) )

    datapath = os.path.join(OUTDIR, "task_spec_data.pkl")
    metadatapath = os.path.join(OUTDIR, "task_spec_metadata.pkl")
    pickle.dump(data, open(datapath,'wb'))
    pickle.dump(metadata, open(metadatapath,'wb'))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task',   required=True, type=str, 
                        help='name of task. DrawerCIP, DoorCIP, ...'),
    parser.add_argument('-p', '--policy', required=True, type=str, 
                        help='path to model.pt file')
    args = parser.parse_args()
    main(args)