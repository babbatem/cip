import json
import itertools
import numpy as np

jobs = json.loads(open('jobs.json','rb').read())
# jobs = json.loads(open('.onager/scripts/CIP-10-28/jobs.json','rb').read())
# print(jobs)
seeds = [0,1,2,3,4]
hyperparameters = {
    "task": ["DoorCIP", "LeverCIP","SlideCIP","DrawerCIP"],
    "grasp": ["False"],
    "seed": [0,1,2,3,4],
    "safety": ["True", "False"],
    "bc": ["True", "False"],
    # "ikstrategy":["random"],
    # "graspstrategy":["None"],
}
keys, values = zip(*hyperparameters.items())
hyperparam_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(hyperparam_permutations)
# for key, val in jobs.items():

to_delete = []
deleted=[]
kept=[]
for permutation in hyperparam_permutations:
    print(permutation)
    seen = False
    strs = [f"{k}_{v}" for k,v in permutation.items()]
    for key, val in jobs.items():
        if np.all([s in val[0] for s in strs]):
            if seen: 
                to_delete.append(key)
                deleted.append(val[0])
                print('DELETING {}'.format(val[0]))
            else:      
                seen = True 
                kept.append(val[0])
                print('SEEN')
                print(val[0])
    # breakpoint()

# print(deleted)

# print(len(np.unique(to_delete)))
print(np.all(["grasp_True" not in d for d in deleted]))
print(np.all(["grasp_False" in d for d in deleted]))
print(np.all(["graspstrategy_None" in k for k in kept]))
print(np.all(["ikstrategy_random" in k for k in kept]))

print(np.all(["ikstrategy_max" not in k for k in kept]))
print(np.all(["graspstrategy_weighted" not in k for k in kept]))

print(to_delete)
print(len(to_delete))
print(len(kept))


# for d in to_delete:
#     print(f"scancel 6459407_{d}")


