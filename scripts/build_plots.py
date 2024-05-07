import os 
import copy
from common import utils
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle 

import pandas as pd 
import seaborn as sns 
sns.set_palette('colorblind')
sns.set(font_scale=1.5)
# sns.set_theme(context='poster', style='whitegrid', palette='colorblind')
sns.set_style('whitegrid', {'axes.grid' : False})
# sns.set( rc = {'axes.labelsize' : 12 })

# LOG_DIR = "./results/final/10-30-CIPs-newUCBs"
# LOG_DIR = "./results/CIP-1-12/"
# LOG_DIR = "./results/CIP-1-12-demo-long"
LOG_DIR = "./results/CIP-1-28/"
# LOG_DIR = "./results/e2e-teachers/"
# LOG_DIR = "./results/e2e-teachers/"

EVAL_FREQ = 10
MAX_EPISODE = 1000
tasks = ["Door", "Slide", "Drawer"]
# tasks = ["Door", "Slide", "Drawer", "Lever"]

mask = {
    "action_scale": 0.5
}

"""
TODOS
- best action scale for each task / args? 
- pad runs to max length
- lines: 
    - grasp false 
    - grasp True ik random grasp random safety false 
    - grasp True ik random grasp random safety false 
    - grasp True ik random grasp ucb    safety true 
    - grasp True ik max    grasp ucb    safety true 

- get screenshots - Figure 1, simulation domains figure 
- add (0,0) to each run!!! 
    PATCH IN AFTER THE FACT?
"""


"""
Door Success        Drawer Success      ...       
Door violation      Drawer violation    ...
...

"""

conditions = { 
                'E2E': 
                    {
                        'grasp' : False,
                        'ik_strategy' : 'random',
                        'grasp_strategy': 'None',
                        'safety': False,
                        'bc': False,
                    },

                'Head':
                    {
                        'grasp' : True,
                        'ik_strategy' : 'random',
                        'grasp_strategy': 'None',
                        'safety': False,
                        'bc': False,
                    },   

                'Safety':
                    {
                        'grasp' : True,
                        'ik_strategy' : 'random',
                        'grasp_strategy': 'None',
                        'safety': True,
                        'bc': False,
                    }, 

                'MV':
                    {
                        'grasp' : True,
                        'ik_strategy' : 'max',
                        'grasp_strategy': 'None',
                        'safety': True,
                        'bc': False,
                    },  

                'CIP':
                    {
                        'grasp' : True,
                        'ik_strategy' : 'max',
                        'grasp_strategy': 'tsr_ucb',
                        'safety': True,
                        'bc': False,
                    },  
                'CIP+BC':
                    {
                        'grasp' : True,
                        'ik_strategy' : 'max',
                        'grasp_strategy': 'tsr_ucb',
                        'safety': True,
                        'bc': True 
                    }, 
                'E2E+BC': 
                    {
                        'grasp' : False,
                        'ik_strategy' : 'random',
                        'grasp_strategy': 'None',
                        'safety': False,
                        'bc': True,
                    }, 
                # 'TEACHER':
                #     {
                #         'grasp' : True,
                #         'ik_strategy' : 'max',
                #         'grasp_strategy': 'None',
                #         'safety': False,
                #         'bc': False, 
                #     },  

            }

def get_data(rootDir, conditions=None, pad_to_length=0):
    scores = {}
    runs=[]
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            desired_fname = 'evaluations.npz'
            if fname == desired_fname:

                try:
                    # print('Found run: %s' % dirName)

                    # if we find a log, read into dataframe
                    path = os.path.join(dirName,fname)

                    eval_files = np.load(open(path,"rb"))
                    eval_successes = eval_files['successes']
                    eval_violations = eval_files['safety_violations']

                    # process the job config
                    config_path = os.path.join(dirName, 'config.pkl')
                    with open(config_path, 'rb') as f:
                        job_data = pickle.load(f)

                    n_episodes = len(eval_successes)*EVAL_FREQ + EVAL_FREQ
                    eval_succ = eval_successes.mean(axis=1)
                    eval_viol = eval_violations.mean(axis=1)
                    
                    # maybe pad to given length 
                    if n_episodes < pad_to_length:
                        n_to_add = int((pad_to_length - n_episodes) / EVAL_FREQ )
                        success_to_add = [eval_succ[-1]]*n_to_add
                        viol_to_add = [eval_viol[-1]]*n_to_add
                        eval_succ = np.append(eval_succ, success_to_add)
                        eval_viol = np.append(eval_viol, viol_to_add)

                    log_df = pd.DataFrame()
                    log_df['success'] = eval_succ
                    log_df['violation'] = eval_viol
                    log_df['episode'] = np.arange(len(log_df))*EVAL_FREQ + EVAL_FREQ
                    log_df['n_episodes'] = n_episodes
                    log_df['tag']=path

                    # print(job_data)
                    for key, val in job_data.items():
                        if key == "task":
                            job_data[key] = job_data[key][:-3]
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

                    # log_df['max_success'] = max_success
                    # log_df['max_violation'] = max_violation
                    log_df['Success Rate'] = max_success
                    log_df['Joint Limit Violation Rate'] = max_violation

                    # maybe filter on conditions 
                    if conditions is not None: 
                        for cond_key, cond_dict in conditions.items(): 
                            if np.all( [log_df[k] == v for k, v in cond_dict.items()] ):
                                log_df['condition'] = cond_key
                                runs.append(log_df)
                                # print(f'found condition {cond_key}')
                                break

                    else:
                        runs.append(log_df)

                            
                except Exception as e:
                    # print(e)
                    raise e
                

    data = pd.concat(runs)
    # print(data.keys())
    return data

# data=get_data(LOG_DIR, conditions=conditions, pad_to_length=MAX_EPISODE)
data=get_data(LOG_DIR, conditions=conditions)
# data=get_data(LOG_DIR)
# breakpoint()
# print(data)

for key, val in mask.items():
    data = data[ data[key] == val ]

tds = []
for task in tasks: 
    
    # grab task 
    cur_data = data[ data['task']==task ]    

    # cut down to minimum episode count
    min_num_episode = np.min(cur_data['n_episodes'])
    ep_mask = cur_data['episode'] < min_num_episode
    cur_data = cur_data[ep_mask]

    # cut to max episodes 
    # ep_mask = cur_data['episode'] < MAX_EPISODE
    # cur_data = cur_data[ep_mask]

    tds.append(cur_data)

data = pd.concat(tds)
og_data = copy.deepcopy(data)
build_conds = []
for condition in conditions:
    build_conds.append(condition)
    data = og_data[ og_data['condition'].isin(build_conds) ]

    for t in ["Success Rate",'Joint Limit Violation Rate']:
        g = sns.relplot(x='episode',
                        y='{t}'.format(t=t),
                        kind='line',
                        data=data,
                        alpha=0.7,
                        hue="condition",
                        # style='seed',
                        hue_order=conditions.keys(),
                        col="task",
                        facet_kws={"sharex":False,"sharey":True},
                        errorbar="se",
                        legend=False,
                        )

        # Adjust the spacing between subplots and legend
        # plt.subplots_adjust(bottom=0.22)

        # sns.move_legend(g, "lower center", bbox_to_anchor=[0.5, -0.1],
        # sns.move_legend(g, "lower center", bbox_to_anchor=[0.45, -0.1],
                    # ncol=len(conditions.keys()), title=None, frameon=False,)

        # g.set(yticklabels=[])
        g.set_titles(col_template = '{col_name}')
        g.set_axis_labels( "Episode" , f"{t}" )
        g.set(ylim=(-0.05, 1.05))
        # plt.savefig(f'{LOG_DIR}/renamed_all_{t}.svg')
        plt.savefig(f'{LOG_DIR}/noleg_build_{t}_{condition}.png', bbox_inches='tight')
        plt.savefig(f'{LOG_DIR}/noleg_build_{t}_{condition}.svg', bbox_inches='tight')
        plt.show()
