import numpy as np
from cpprb import ReplayBuffer, PrioritizedReplayBuffer, create_env_dict, create_before_add_func
import random


class buffer_class:
	def __init__(self, max_length, seed_number, env, params):
		env_dict = create_env_dict(env)
		self.before_add = create_before_add_func(env)
		self.params = params

		# used to schedule beta (compensation param)
		self.call_counter = 0
		self.increments = 10 # increments for beta
		self.move_up_every = int((self.params["max_episode"])/(self.increments))
		if self.params['per']:
			self.beta_schedule = np.linspace(self.params['per_beta_start'], 1.0, self.increments)
			self.current_beta_index = -1

		if params['per']:
			if params['nstep']:
				self.storage = PrioritizedReplayBuffer(max_length, env_dict, Nstep={
				"size": params["nstep_size"],
				"gamma": params["gamma"],
				"rew": "rew",
				"next": "next_obs"
			}, alpha=params['alpha'])
			else:
				self.storage = PrioritizedReplayBuffer(max_length, env_dict, alpha=params['alpha'])
		else:
			if params['nstep']:
				self.storage = ReplayBuffer(max_length, env_dict, Nstep={
					"size": params["nstep_size"],
					"gamma": params["gamma"],
					"rew": "rew",
					"next": "next_obs"
				})
			else:
				self.storage = ReplayBuffer(max_length, env_dict)

	def append(self, s, a, r, done, sp):
		if self.params['nstep']:
			# breakpoint()
			# s = s.reshape(1, s.shape[0])
			# a = a.reshape(1, a.shape[0])
			# sp = sp.reshape(1, sp.shape[0])
			if not done:
				self.storage.add(**self.before_add(obs=s,
					act=a,
					rew=r,
					next_obs=sp,
					done=0.0))
			else:
				self.storage.on_episode_end()
		else:
			self.storage.add(**self.before_add(obs=s, act=a, rew=r, done=done, next_obs=sp))

	def sample(self, batch_size):
		if self.params['per']:
			if (self.params['should_schedule_beta'] and 
			self.call_counter % (self.move_up_every*self.params['updates_per_episode']) == 0
			and (self.call_counter != 0 or self.current_beta_index == -1)):
				self.current_beta_index += 1
				print("Switching to PER beta of:", self.beta_schedule[min(self.current_beta_index, len(self.beta_schedule) - 1)])
				self.call_counter = 0 # prevent overflow

			self.call_counter += 1
			batch = self.storage.sample(batch_size, self.beta_schedule[min(self.current_beta_index, len(self.beta_schedule) - 1)])

			s_matrix = batch['obs']
			a_matrix = batch['act']
			r_matrix = batch['rew']
			done_matrix = batch['done']
			sp_matrix = batch['next_obs']
			weights = batch['weights']
			indexes = batch['indexes']
			return s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix, weights, indexes
		else:
			batch = self.storage.sample(batch_size)
			s_matrix = batch['obs']
			a_matrix = batch['act']
			r_matrix = batch['rew']
			done_matrix = batch['done']
			sp_matrix = batch['next_obs']
			return s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix

	def __len__(self):
		return self.storage.get_stored_size()
