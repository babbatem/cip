import os 
import numpy as np 
import pickle
import matplotlib.pyplot as plt		

import torch  
from torch.utils.data import Dataset

class TaskSuccessDataset(Dataset):
	"""docstring for TaskSuccessDataset"""
	def __init__(self, path):
		super(TaskSuccessDataset, self).__init__()
		self.path = path
		with open(self.path,'rb') as f: 
			raw_data = pickle.load(f)

		# (grasp, test_qpos, w, p, wp, states, actions, info['success'])
		max_traj_len = -np.inf
		for datum in raw_data:
			states = datum[5]
			actions = datum[6]
			state_shape = len(states[0])
			action_shape = len(actions[0])
			if len(states) > max_traj_len:
				max_traj_len = len(states)

		self.grasps = np.zeros((len(raw_data), 4, 4))
		self.qpos   = np.zeros((len(raw_data), 7))
		self.manips = np.zeros((len(raw_data), 3))
		self.states = np.zeros((len(raw_data), max_traj_len, state_shape))
		self.labels = np.zeros((len(raw_data)))
		
		for i, datum in enumerate(raw_data):
			states = datum[5]
			traj_len = len(states)

			self.grasps[i] = datum[0]
			self.qpos[i] = datum[1]	 
			self.manips[i] = datum[4]
			self.states[i,:traj_len] = datum[5]
			self.labels[i] = datum[-1]

		# to torch 
		self.device = torch.device('cuda')
		self.grasps = torch.from_numpy(self.grasps).to(self.device)
		self.qpos   = torch.from_numpy(self.qpos  ).to(self.device)
		self.manips = torch.from_numpy(self.manips).to(self.device)
		self.states = torch.from_numpy(self.states).to(self.device)
		self.labels = torch.from_numpy(self.labels).to(self.device)

		print(self.grasps.dtype)
		print(self.qpos.dtype)
		print(self.manips.dtype)
		print(self.states.dtype)
		print('------')
		print(self.labels.dtype)
		print(self.labels.max())
		print(self.labels.min())

		# TODO: generate to get more grasps. 30 probably aint enough. 
		# TODO: generate grasp xyz, quat and concat to get input vectors 
		# TODO: test-set...hmm...second dataset? 

	def __len__(self):
		return len(self.grasps)

	def __getitem__(self, idx):
		x = self.grasps

	def vis_stats(self):
		plt.hist(self.labels.cpu().numpy())
		plt.show()

		dirname = os.path.dirname(self.path)
		metadatapath = os.path.join(dirname, "task_spec_metadata.pkl")
		with open(metadatapath,'rb') as f: 
			meta_data = pickle.load(f)
		
		gs = []
		success = []
		unsafe = []
		for i, md in enumerate(meta_data):
			gs.append(md[0])
			success.append(md[1])
			unsafe.append(md[2])
		plt.hist(success)
		plt.show()



class TrajectoryDataset(TaskSuccessDataset):
	"""docstring for TaskSuccessDataset"""
	def __init__(self, path):
		super(TrajectoryDataset, self).__init__(path)

	def __len__(self):
		pass 
	
	def __getitem__(self, idx):
		pass 


		
if __name__ == '__main__':
	path = "/home/abba/motor_skills/task_spec/grasp_none_ik_random/task_spec_data.pkl"
	dataset = TaskSuccessDataset(path)
	dataset.vis_stats()