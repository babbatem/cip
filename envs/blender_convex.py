import bpy 
import os

folder_path = "/Users/abba/projects/motor_skills/robosuite/robosuite/models/assets/objects/meshes/40147/textured_objs"
for root, dirs, files in os.walk(folder_path, topdown=False):
	for name in files:
		bpy.ops.wm.read_factory_settings(use_empty=True)
		if ("convex" in name) and (".obj" in name):
			full_name = os.path.join(folder_path, name)
			result = bpy.ops.import_scene.obj(filepath=full_name)
			print(full_name)
			print(result)

			new_name = full_name[:-4] + ".stl"
			result = bpy.ops.export_mesh.stl(filepath=new_name)
			print(new_name)
			print(result)

