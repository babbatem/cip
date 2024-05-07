from common import utils
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


"""
log_dir = "./DoorCIPFinal"
eval_freq = 10
tasks = ["DoorCIP"]#"LeverCIP","SlideCIP","DrawerCIP"]
seeds = [0,1,2,3,4,5,6,7,8,9,10]
grasp_safety_bc = [[False, True, False],
				[True, True, False],
				[True, True, True]]


"""
log_dir = "./SlideCIPChecker"
eval_freq = 10
tasks = ["SlideCIP"]#"LeverCIP","SlideCIP","DrawerCIP"]
seeds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
grasp_safety_bc = [[False, True, False],
				[True, True, False],
				[True, True, True]]


g_s_b_map = {str([False, True, False]): "No head + No behavioral cloning",
str([True, True, False]): "head + No behavioral cloning",
str([True, True, True]): "head + behavioral cloning"}

def return_evaluations(dir_name):
	task_dirs = [f for f in listdir(log_dir)]
	for task_dir in task_dirs:
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
	for g_s_b in grasp_safety_bc:
		success_seeds = []
		violation_seeds = []
		shortest_episodes = 1000000000000000000000000
		print(g_s_b)
		for seed in seeds:
			try:
				dir_name = "seed_{}__task_{}__grasp_{}__safety_{}__bc_{}".format(seed,task,g_s_b[0],g_s_b[1],g_s_b[2])
				#dir_name = "_grasp_{grasp}__safety_{safety}__bc_{bc}".format(grasp=g_s_b[0], safety=g_s_b[1], bc=g_s_b[2])
				average_success, average_violation = return_evaluations(dir_name)
				print(len(average_success))
				if len(average_success) < shortest_episodes:
					shortest_episodes = len(average_success)
				success_seeds.append(average_success)
				violation_seeds.append(average_violation)
			except:
				print(dir_name +" failed")

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

		success_task_dict[str(g_s_b)] = [np_success_averages,np_success_stds]
		violation_task_dict[str(g_s_b)] = [np_violation_averages,np_violation_stds]

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
			


	for key in list(success_task_dict.keys()):
		x_axis = list(range(len(success_task_dict[key][0])))
		x_axis = [x * eval_freq for x in x_axis]
		plt.errorbar(x_axis, success_task_dict[key][0], success_task_dict[key][1], label=g_s_b_map[key])

	leg = plt.legend(loc='upper left')
	plt.xlabel('Number of training episodes') 
	plt.ylabel('Task success rate') 
	plt.title("Task success rate vs training episodes ({})".format(task[:-3]))
	plt.ylim([-0.01, 1.01])
	if task =="SlideCIP":
		plt.xlim([0,50])
	plt.savefig(log_dir+"/"+task+"_success.png")
	plt.show()

	for key in list(violation_task_dict.keys()):
		x_axis = list(range(len(violation_task_dict[key][0])))
		x_axis = [x * eval_freq for x in x_axis]
		plt.errorbar(x_axis, violation_task_dict[key][0], violation_task_dict[key][1], label=g_s_b_map[key])

	leg = plt.legend(loc='upper right')
	plt.xlabel('Number of training episodes') 
	plt.ylabel('Safety violation rate') 
	plt.title("Safety violation rate vs training episodes ({})".format(task[:-3]))
	plt.ylim([-0.01, 1.01])
	plt.savefig(log_dir+"/"+task+"_safety.png")
	plt.show()




