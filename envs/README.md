# Converting models to mujoco format 
1. convex decomposition to fix the 2D meshes: pb_convex_approx.py (hardcoded filepath, beware) 
2. convert .obj to .stl: run blender_convex.py through blender (hardcoded filepath again)
3. copy the resulting .stl files from textured_objs folder up one level to object root 
4. todo: edit the xml such that joint function properly
5. ./$HOME/.mujoco/mujoco210/bin/simulate /path/to/xml 