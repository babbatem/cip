#!/bin/bash
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 12:00:00

python ppo_baselines.py
