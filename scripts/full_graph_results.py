from common import utils
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import itertools


log_dir = "./CIP-10-28"
eval_freq = 10
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

keys, values = zip(*hyperparameters.items())
hyperparam_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]


def return_evaluations(dir_name):
	task_dirs = [f for f in listdir(log_dir)]
	for task_dir in task_dirs:
		# print(task_dir)
		# print(dir_name)
		# breakpoint()
		if dir_name in task_dir:
			path_to_eval = log_dir+"/"+task_dir+"/evaluations.npz"

	eval_files = np.load(open(path_to_eval,"rb"))

	eval_successes = eval_files['successes']
	eval_violations = eval_files['safety_violations']

	average_successes = []
	average_violations = []

	for eval_episode in eval_successes:
		success_ints = [1 if e == True else 0 for e in eval_episode]
		average_successes.append(np.mean(success_ints))
	for eval_episode in eval_violations:
		violation_ints = [1 if e == True else 0 for e in eval_episode]
		average_violations.append(np.mean(violation_ints))

	return(average_successes,average_violations)

#success
for task in tasks:
	success_task_dict = {} #task_dict[grasp_safety_bc][seed]
	violation_task_dict = {} #task_dict[grasp_safety_bc][seed]
	for permutation in hyperparam_permutations:
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

				#dir_name = "_grasp_{grasp}__safety_{safety}__bc_{bc}".format(grasp=g_s_b[0], safety=g_s_b[1], bc=g_s_b[2])
				average_success, average_violation = return_evaluations(dir_name)
				print(len(average_success))
				if len(average_success) < shortest_episodes:
					shortest_episodes = len(average_success)
				success_seeds.append(average_success)
				violation_seeds.append(average_violation)
			except Exception as e:
				print(dir_name +" failed")
				# raise e

		shortened_success = []
		shortened_violations = []
		for success_seed in success_seeds:
			shortened_success.append(np.array(success_seed[:shortest_episodes]))
		for violation_seed in violation_seeds:
			shortened_violations.append(np.array(violation_seed[:shortest_episodes]))

		np_shortened_success = np.array(shortened_success)
		np_shortened_violation = np.array(shortened_violations)

		np_success_averages = np.average(np_shortened_success,axis=0)
		np_success_stds = stats.sem(np_shortened_success,axis=0)
		np_violation_averages = np.average(np_shortened_violation,axis=0)
		np_violation_stds = stats.sem(np_shortened_violation,axis=0)

		success_task_dict[dir_name_no_seed] = [np_success_averages,np_success_stds]
		violation_task_dict[dir_name_no_seed] = [np_violation_averages,np_violation_stds]

	try:
		print("Shorted cutting now!")
		#shorten success and violations based on shortest ablations
		shortest_runs = 1000000000000000000000
		for key in list(success_task_dict.keys()):
			num_runs = len(success_task_dict[key][0])
			print(num_runs)
			if num_runs < shortest_runs:
				shortest_runs = num_runs
		print("Shortest runs:",shortest_runs)
		for key in list(success_task_dict.keys()):
			success_task_dict[key][0] = success_task_dict[key][0][:shortest_runs]
			success_task_dict[key][1] = success_task_dict[key][1][:shortest_runs]

			violation_task_dict[key][0] = violation_task_dict[key][0][:shortest_runs]
			violation_task_dict[key][1] = violation_task_dict[key][1][:shortest_runs]
	except Exception as e:
		print(e)
			


	for key in list(success_task_dict.keys()):
		try:
			plt.subplot(1, 2, 1)
			x_axis = list(range(len(success_task_dict[key][0])))
			x_axis = [x * eval_freq for x in x_axis]
			mean_1 = success_task_dict[key][0]
			std_1 = success_task_dict[key][1]
			plt.plot(x_axis, mean_1,label=key)
			plt.fill_between(x_axis, mean_1 - std_1, mean_1 + std_1, alpha=0.2)

			leg = plt.legend(loc='lower left')
			plt.xlabel('Number of training episodes') 
			plt.ylabel('Task success rate') 
			plt.title("Task success rate vs training episodes ({})".format(task[:-3]))
			plt.ylim([-0.01, 1.01])

			x_axis = list(range(len(violation_task_dict[key][0])))
			x_axis = [x * eval_freq for x in x_axis]
			plt.subplot(1, 2, 2)
			mean_1 = violation_task_dict[key][0]
			std_1 = violation_task_dict[key][1]
			plt.plot(x_axis, mean_1)
			plt.fill_between(x_axis, mean_1 - std_1, mean_1 + std_1, alpha=0.2)

			#leg = plt.legend(loc='upper left')
			plt.xlabel('Number of training episodes') 
			plt.ylabel('Violation rate') 
			plt.title("Violation rate vs training episodes ({})".format(task[:-3]))
			plt.ylim([-0.01, 1.01])
		except Exception as e:
			print(e)
	
	plt.savefig(log_dir+"/"+task+"_success_violation.png")	
	plt.show()

		
		


