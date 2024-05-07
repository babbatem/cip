import ppo_baselines

training_steps = [500000]

config = {}
for training_step in training_steps:
	config["training_steps"] = training_step
	ppo_baselines.train_robosuite(config)
