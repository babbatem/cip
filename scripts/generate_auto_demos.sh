#!/bin/bash

N_DEMOS=100

# venv
source ~/envs/cip/bin/activate

# delete existing demos 
rm -f ./auto_demos/*/*

# for each task, regenerate
python trained_agent_demos.py -t DoorCIP \
                              --grasp \
                              --guarantee \
                              --num_episodes ${N_DEMOS} \
                              --log_path results/CIP-1-12-demo-long/CIP-1-12-demo-long_1__experimentname_CIP-1-12-demo-long__seed_0__task_DoorCIP__grasp_True__bc_False__safety_False__graspstrategy_None__ikstrategy_max__actionscale_0.5__noisestd_0.05__lr_0.0001__batchsize_256__gradientsteps_-1__maxevaleps_2000/
                              

python trained_agent_demos.py -t DrawerCIP \
                              --grasp \
                              --guarantee \
                              --num_episodes ${N_DEMOS} \
                              --log_path results/CIP-1-12-demo-long/CIP-1-12-demo-long_4__experimentname_CIP-1-12-demo-long__seed_1__task_DrawerCIP__grasp_True__bc_False__safety_False__graspstrategy_None__ikstrategy_max__actionscale_0.5__noisestd_0.05__lr_0.0001__batchsize_256__gradientsteps_-1__maxevaleps_2000


python trained_agent_demos.py -t SlideCIP \
                              --grasp \
                              --guarantee \
                              --num_episodes ${N_DEMOS} \
                              --log_path results/CIP-1-12-demo-long/CIP-1-12-demo-long_5__experimentname_CIP-1-12-demo-long__seed_0__task_SlideCIP__grasp_True__bc_False__safety_False__graspstrategy_None__ikstrategy_max__actionscale_0.5__noisestd_0.05__lr_0.0001__batchsize_256__gradientsteps_-1__maxevaleps_2000/                    


python trained_agent_demos.py -t LeverCIP \
                              --grasp \
                              --guarantee \
                              --num_episodes ${N_DEMOS} \
                              --log_path results/CIP-1-12-demo-long/CIP-1-12-demo-long_7__experimentname_CIP-1-12-demo-long__seed_0__task_LeverCIP__grasp_True__bc_False__safety_False__graspstrategy_None__ikstrategy_max__actionscale_0.5__noisestd_0.05__lr_0.0001__batchsize_256__gradientsteps_-1__maxevaleps_2000/



