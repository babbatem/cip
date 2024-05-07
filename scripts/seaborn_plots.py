import os 
from common import utils
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle 

import pandas as pd 
import seaborn as sns 
sns.set_palette('colorblind')


LOG_DIR = "./10-30-CIPs-newUCBs"
EVAL_FREQ = 10
tasks = ["DoorCIP", "LeverCIP","SlideCIP","DrawerCIP"]
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

def get_data(rootDir):
    scores = {}
    runs=[]
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            desired_fname = 'evaluations.npz'
            if fname == desired_fname:

                try:
                    print('Found run: %s' % dirName)

                    # if we find a log, read into dataframe
                    path = os.path.join(dirName,fname)

                    eval_files = np.load(open(path,"rb"))
                    eval_successes = eval_files['successes']
                    eval_violations = eval_files['safety_violations']

                    # process the job config
                    config_path = os.path.join(dirName, 'config.pkl')
                    with open(config_path, 'rb') as f:
                        job_data = pickle.load(f)

                    log_df = pd.DataFrame()
                    log_df['success'] = eval_successes.mean(axis=1)
                    log_df['violation'] = eval_violations.mean(axis=1)
                    log_df['episode'] = np.arange(len(log_df))*EVAL_FREQ
                    log_df['n_episodes'] = len(log_df)*EVAL_FREQ
                    log_df['tag']=path

                    print(job_data)
                    for key, val in job_data.items():
                        log_df[key] = job_data[key]
                    
                    # log_df['max_success'] = np.maximum.accumulate(log_df['success'])
                    # log_df['max_success'] = np.maximum.accumulate(log_df['success'])
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
                    runs.append(log_df)
                except Exception as e:
                    print(e)
                

    data = pd.concat(runs)
    # print(data.keys())
    return data

data=get_data(LOG_DIR)
print(data)

for task in tasks: 
    cur_data = data[ data['task']==task ]

    # my_relplot = sns.relplot(x='episode',
    #                          y='max_success',
    #                          kind='line',
    #                          data=cur_data,
    #                          alpha=0.4,
    #                          hue='grasp',
    #                          size="grasp_strategy",
    #                          style="ik_strategy",
    #                          row="safety",
    #                          col="bc"
    #                          )
    for t in ["success","violation"]:
        my_relplot = sns.relplot(x='episode',
                                 y='max_{t}'.format(t=t),
                                 kind='line',
                                 data=cur_data,
                                 alpha=0.4,
                                 hue="grasp_strategy",
                                 style="ik_strategy",
                                 row="safety",
                                 col="action_scale"
                                 )

        plt.suptitle(task)
        plt.savefig(f'{LOG_DIR}/{task}_{t}.svg')
        plt.show()
