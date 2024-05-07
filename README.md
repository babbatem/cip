# Composable Interaction Primitives (CIPs)
### A Structured Policy Class for Efficiently Learning Sustained-Contact Manipulation Skills (ICRA 2024)
#### [Ben Abbatematteo*](https://babbatem.github.io/), [Eric Rosen*](https://eric-rosen.github.io/), Skye Thompson, Tuluhan Akbulut, Sreehari Rammohan, George Konidaris
#### Intelligent Robot Lab, Brown University 
[[Paper]](https://cs.brown.edu/people/gdk/pubs/cips.pdf) [[Video]](https://www.youtube.com/watch?v=f-R4Sz8HohU&t=14s&ab_channel=IRLLab)

```
@inproceedings{abbatematteo2024composable,
  title={Composable Interaction Primitives: A Structured Policy Class for Efficiently Learning Sustained-Contact Manipulation Skills},
  author={Abbatematteo, Ben and Rosen, Eric and Thompson, Skye and Akbulut, Tuluhan and Rammohan, Sreehari and Konidaris, George},
  booktitle={Proceedings of the 2024 IEEE Conference on Robotics and Automation},
  year={2024},
  organization={Proceedings of the 2024 IEEE Conference on Robotics and Automation}
}
```

### Install (Slurm)
```

  # setup modules, virtual environment
  module load opengl/mesa-12.0.6 python/3.9.0
  python3 -m venv ~/envs/cip
  source $HOME/envs/cip/bin/activate	

  # clone 
  git clone git@github.com:babbatem/motor_skills.git    
  git submodule init    
  git submodule update

  # assuming ~/.mujoco 
  export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH

  cd ~/motor_skills/
  pip install -r requirements.txt 
  pip install -e third_party/robosuite

  # test with
  python -c "import mujoco_py"
  python -c "import robosuite"

  # IKFlow setup
  git lfs pull origin task_specific # Download pretrained models

```

### Running Jobs with Onager
First, set the slurm configuration using the provided script (double check mujoco, virtualenv paths)
```
  ./write_slurm_header.sh
```
Then, for example: 
```
  onager prelaunch +jobname exp1 +command "python scripts/run_ppo_baseline.py " +arg --experiment_name exp1 +arg --seed 0 1 2 +arg --task DoorCIP +arg --grasp True False +tag --run_title 
  
  onager launch --backend slurm --jobname exp1 --duration 8:00:00 --gpus 1 --mem 44 --partition 3090-gcondo
```

### Visualize a learned policy and evaluate
Make sure that the saved logs are inside ./logs in the base folder. Then, you can run:

```
python scripts/vis_policy.py --task [TASK] --grasp [BOOLEAN] --safety [BOOLEAN] --num_episodes [INT[]]
```

For example:
```
python scripts/vis_policy.py --task DrawerCIP --grasp True --safety True --num_episodes 3
```


### Regenerating grasps for robosuite tasks
In order to get good grasps, we first need to get a clean point cloud of the object of interest to grasp that is in the world frame of robosuite. You can run the following command to do so:

```
python scripts/task2handlePC.py --task [TASK]
```

where [TASK] is the name of your robosuite task (e.g DoorCIP, DrawerCIP, LeverCIP, SlideCIP). You can also pass in the list of cameras you want to build the point cloud from, as well whether debug is true or false, and the element id. This code will render images from the cameras, get the RGB-D images from each camera, and segment out the object of interest based on the element id. The element id of the object of interest is specifc to each environment, and so in order to get it, you can pass in True to debug, and look at the bottom image (The top is RGB and middle is depth), which gives you the element id of each object segment in the image. Hover your mouse over the object of interest, and you'll see the integer value in the bottom right (For example, in DoorCIP, the handle is 53). Now that you know the element id, you can pass it in as an argument, and all the masked point clouds from each camera will automatically be merged (it is also probably worth adding the element id for the task to the top of the file in a comment, so others can easily identifty it). For example, on DoorCIP, you could run:

```
python scripts/task2handlePC.py --task DoorCIP --element_id 53 --cameras birdview agentview frontview
```

This will save the merged point clouds into the ```pointcloud``` directory as a ply based on the name of the task. If you want to visualize the point cloud to confirm it makes sense, you can run

```
python scripts/point_cloud_viz.py --task [TASK]
```

This will visualize the merged point cloud in the world frame (you will see a giant 3D frame in the scene along with the point cloud, where red, green, blue reprsesen xyz in robosuite respectively). You can move around the scene and confirm the point cloud is good.

Now that you have confirmed you have a good, segmented point cloud in world frame, we can generate grasps for the object. You can run the following command:

```
python scripts/GraspSelector.py --task [TASK] --num_samples [NUMBER]
```
Here, num_samples is the number of samples the grasp heuristic will generate, visualize and save. When you run this, you should see the same scene as you did before in ```point_cloud_viz```, except now there will be sampled frames on the point cloud representing grasp poses. As the code indicates in a print statement: Note that gripper will approach along posive z axis (blue), and align parallel gripper with x axis (red). In general, blue axis should be pointing into the object, and red axis should be aligned with the direction of curvature. When you exist the visualization, the grasp poses will be saved into the ```grasp``` folder as a .pkl, with the file name being the name of the task. Note that each pose is represented by a 4x4 matrix, where the top left 3x3 represent the rotation matrix, and the first 3 rows of the far-right column represent the position.

Some of the grasps we get from GraspSelector are not great for running the CIP because we get infeasible IK solutions for motion planning or the gripper is contact with geometries in the scene when we go to that pose. To filter out bad poses, you can run:

```
python scripts/filter_cip_grasps.py --task [TASK]
``` 
Note that this will overwrite the same pickle file it loads from, so note that if you run GraspSelector.py again, you will need to run this again.


And that's it! Now when you run
```
python scripts/run_ppo_baseline.py --task [TASK] --grasp True
```

The robot should be using the grasp heuristic!


Note: If you get the error:
```RuntimeError: Failed to initialize OpenGL```
You need to run the command:
```
unset LD_PRELOAD
```

