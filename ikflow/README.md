# IKFlow
Normalizing flows for Inverse Kinematics. Open source implementation to the paper ["IKFlow: Generating Diverse Inverse Kinematics Solutions"](https://ieeexplore.ieee.org/abstract/document/9793576)

[![arxiv.org](https://img.shields.io/badge/cs.RO-%09arXiv%3A2111.08933-red)](https://arxiv.org/abs/2111.08933)

See https://github.com/jstmn/ikflow for further details


## Getting started

Evaluate a pretrained IKFlow model with the following command. Note that `model_name` should be in 'ikflow/model_descriptions.yaml'
```
python ikflow/evaluate.py \
    --samples_per_pose=50 \
    --testset_size=500 \
    --n_samples_for_runtime=512 \
    --model_name=panda_lite
```

## Common errors

1. Pickle error when loading a pretrained model from gitlfs. This happens when the pretrained models haven't been downloaded by gitlfs. For example, they may be 134 bytes (they should be 100+ mb). Run `git lfs pull origin master` 
```
Traceback (most recent call last):
  File "evaluate.py", line 117, in <module>
    ik_solver, hyper_parameters = get_ik_solver(model_weights_filepath, robot_name, hparams)
  File "/home/jstm/Projects/ikflow/utils/utils.py", line 55, in get_ik_solver
    ik_solver.load_state_dict(model_weights_filepath)
  File "/home/jstm/Projects/ikflow/utils/ik_solvers.py", line 278, in load_state_dict
    self.nn_model.load_state_dict(pickle.load(f))
_pickle.UnpicklingError: invalid load key, 'v'.
```

## TODO
1. [ ] Solution refinement - use an existing IK solver to refine solutions
2. [ ] Solution refinement - Implement and include a parallelized IK solver   
