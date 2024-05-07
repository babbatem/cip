#!/bin/bash

N_DEMOS=10

# venv
source ~/envs/cip/bin/activate

# delete existing demos 
rm -f ./e2e_auto_demos/*/*

# for each task, regenerate
python trained_agent_demos.py -t DoorCIP \
                              --guarantee \
                              --num_episodes ${N_DEMOS} \
                              --log_path  ../og_motor_skills/results/e2e-teachers/e2e-teachers_04__experimentname_e2e-teachers__seed_4__task_DoorCIP__grasp_False__bc_False__safety_False__graspstrategy_None__ikstrategy_random__actionscale_0.5__noisestd_0.05__lr_0.0001__nevalepisodes_20__maxevaleps_20000/
                              

python trained_agent_demos.py -t DrawerCIP \
                              --guarantee \
                              --num_episodes ${N_DEMOS} \
                              --log_path  ../og_motor_skills/results/e2e-teachers/e2e-teachers_09__experimentname_e2e-teachers__seed_4__task_DrawerCIP__grasp_False__bc_False__safety_False__graspstrategy_None__ikstrategy_random__actionscale_0.5__noisestd_0.05__lr_0.0001__nevalepisodes_20__maxevaleps_20000/


python trained_agent_demos.py -t SlideCIP \
                              --guarantee \
                              --num_episodes ${N_DEMOS} \
                              --log_path ../og_motor_skills/results/e2e-teachers/e2e-teachers_13__experimentname_e2e-teachers__seed_3__task_SlideCIP__grasp_False__bc_False__safety_False__graspstrategy_None__ikstrategy_random__actionscale_0.5__noisestd_0.05__lr_0.0001__nevalepisodes_20__maxevaleps_20000/  


python trained_agent_demos.py -t LeverCIP \
                              --guarantee \
                              --num_episodes ${N_DEMOS} \
                              --log_path ../og_motor_skills/results/e2e-teachers/e2e-teachers_18__experimentname_e2e-teachers__seed_3__task_LeverCIP__grasp_False__bc_False__safety_False__graspstrategy_None__ikstrategy_random__actionscale_0.5__noisestd_0.05__lr_0.0001__nevalepisodes_20__maxevaleps_20000/


                              



