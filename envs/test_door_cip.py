if __name__ == '__main__':
	import robosuite
	from robosuite.controllers import load_controller_config

	# load default controller parameters for Operational Space Control (OSC)
	controller_config = load_controller_config(default_controller="OSC_POSE")

	# create an environment to visualize on-screen
	env = robosuite.make(
		    "DoorCIP",
		    robots=["Panda"],             # load a Sawyer robot and a Panda robot
		    gripper_types="default",                # use default grippers per robot arm
		    controller_configs=controller_config,   # each arm is controlled using OSC
		    env_configuration="single-arm-opposed", # (two-arm envs only) arms face each other
		    has_renderer=True,                      # on-screen rendering
		    render_camera="frontview",              # visualize the "frontview" camera
		    has_offscreen_renderer=False,           # no off-screen rendering
		    control_freq=20,                        # 20 hz control for applied actions
		    horizon=200,                            # each episode terminates after 200 steps
		    use_object_obs=False,                   # no observations needed
		    use_camera_obs=False,                   # no observations needed
		)

	# this example assumes an env has already been created, and performs one agent rollout
	import numpy as np

	def get_policy_action(obs):
	    # a trained policy could be used here, but we choose a random action
	    low, high = env.action_spec
	    return np.random.uniform(low, high)*0.0 

	while True:

		# reset the environment to prepare for a rollout
		obs = env.reset()

		done = False
		ret = 0.
		while not done:
		    action = get_policy_action(obs)         # use observation to decide on an action
		    obs, reward, done, _ = env.step(action) # play action
		    env.render()
		    ret += reward
		print("rollout completed with return {}".format(ret))

