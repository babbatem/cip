from common import utils
from os import listdir
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd 
import seaborn as sns
import pickle

import itertools


LOG_DIR = "./DEV"
EVAL_FREQ = 10
tasks = ["DoorCIP", "LeverCIP","SlideCIP","DrawerCIP"]
# tasks=["DrawerCIP"]
seeds = [0,1,2,3,4]

hyperparameters = {
    "grasp": ["True","False"],
    "safety": ["True", "False"],
    "bc": ["True", "False"],
    "noisestd": ["0.05"],
    "lr": ["0.0001"],
    "batchsize": ["256"],
    "gradientsteps": ["-1"],
    "graspstrategy":["None", "weighted"],
    "ikstrategy":["random","max"],
    "actionscale":["1.0"]
}

keys, values = zip(*hyperparameters.items())
hyperparam_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]


def return_evaluations(dir_name, permutation):
    task_dirs = [f for f in listdir(LOG_DIR)]
    for task_dir in task_dirs:
        # print(task_dir)
        # print(dir_name)
        # breakpoint()

        # print(dir_name)
        # print(task_dir)
        # breakpoint()

        if dir_name in task_dir:
            path_to_eval = LOG_DIR+"/"+task_dir+"/evaluations.npz"

    eval_files = np.load(open(path_to_eval,"rb"))

    config_path = os.path.join(LOG_DIR+"/"+task_dir, 'config.pkl')
    print(f'writing {config_path}')
    with open(config_path, 'wb') as f:
        pickle.dump(permutation, f)

    eval_successes = eval_files['successes']
    eval_violations = eval_files['safety_violations']
    
    log_df = pd.DataFrame()
    log_df['success'] = eval_successes.mean(axis=1)
    log_df['violation'] = eval_violations.mean(axis=1)
    log_df['episode']=np.arange(len(log_df))*EVAL_FREQ
    log_df['tag']=path_to_eval
    for key, val in permutation.items():
        log_df[key] = permutation[key]

    max_success = []
    max_violation = []
    best_succ = -1
    best_succ_violations = -1
    for i, succ in enumerate(log_df['success']):
        if succ > best_succ:
            best_succ=succ
            best_succ_violations=log_df['violation'][i]
        max_success.append(best_succ)
        max_violation.append(best_succ_violations)

    log_df['max_success'] = max_success
    log_df['max_violation'] = max_violation
    return log_df

#success
runs=[]
for task in tasks:
    success_task_dict = {} #task_dict[grasp_safety_bc][seed]
    violation_task_dict = {} #task_dict[grasp_safety_bc][seed]
    min_length = np.inf
    for permutation in hyperparam_permutations:
        permutation['task']=task
        print(permutation)
        success_seeds = []
        violation_seeds = []
        shortest_episodes = 1000000000000000000000000
        dir_name_no_seed = "task_{task}__grasp_{grasp}__bc_{bc}__safety_{safety}__graspstrategy_{graspstrategy}__ikstrategy_{ikstrategy}__actionscale_{actionscale}__noisestd_{noisestd}__lr_{lr}__batchsize_{batchsize}__gradientsteps_{gradientsteps}".format(
                    task=task,
                    grasp=permutation["grasp"],
                    bc=permutation["bc"],
                    safety=permutation["safety"],
                    graspstrategy=permutation["graspstrategy"],
                    ikstrategy=permutation["ikstrategy"],
                    actionscale=permutation["actionscale"],
                    noisestd=permutation["noisestd"],
                    lr=permutation["lr"],
                    batchsize=permutation["batchsize"],
                    gradientsteps=permutation["gradientsteps"])
        for seed in seeds:
            try:
                dir_name = "seed_{seed}__task_{task}__grasp_{grasp}__bc_{bc}__safety_{safety}__graspstrategy_{graspstrategy}__ikstrategy_{ikstrategy}__actionscale_{actionscale}__noisestd_{noisestd}__lr_{lr}__batchsize_{batchsize}__gradientsteps_{gradientsteps}".format(
                    seed=str(seed),
                    task=task,
                    grasp=permutation["grasp"],
                    bc=permutation["bc"],
                    safety=permutation["safety"],
                    graspstrategy=permutation["graspstrategy"],
                    ikstrategy=permutation["ikstrategy"],
                    actionscale=permutation["actionscale"],
                    noisestd=permutation["noisestd"],
                    lr=permutation["lr"],
                    batchsize=permutation["batchsize"],
                    gradientsteps=permutation["gradientsteps"])
                
                run_df = return_evaluations(dir_name, permutation)
                length = len(run_df)
                if length < min_length:
                    min_length = length 
                runs.append(run_df)
                
            except Exception as e:
                print(dir_name +" failed")
                # raise e
                print(e)

      
    # data = pd.concat(runs)
    # data = data[ data['episode'] < 1000 ]
    # print(data.keys())
    # my_relplot = sns.relplot(x='episode',
    #                          y='max_success',
    #                          kind='line',
    #                          data=data,
    #                          alpha=0.4,
    #                          hue='grasp',
    #                          size="graspstrategy",
    #                          style="ikstrategy",
    #                          row="safety",
    #                          col="bc"
    #                          )
    
    # plt.suptitle(task)
    # plt.savefig(f'LOG_DIR+{task}_shortened.svg')
    # plt.show()


