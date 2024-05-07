import pybullet as p
import os 

p.connect(p.DIRECT)
folder_path = "/Users/abba/projects/motor_skills/robosuite/robosuite/models/assets/objects/meshes/40147/textured_objs"

for root, dirs, files in os.walk(folder_path):
	for name in files:
		if ("original" in name) and (".obj" in name):
			full_name = os.path.join(folder_path, name)
			name_in = full_name
			name_out = full_name.replace("original", "convex")
			name_log = full_name.replace("original", "log")
			p.vhacd(name_in, name_out, name_log, alpha=0.04,resolution=50000)