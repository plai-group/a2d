#!/usr/bin/env bash

# launch_multiple.sh
# Wrap launching the main gridworld experiments.
# This script will repeatedly call the SLURM job script _launch_multiple.sh
# This outer script sets the folder into which `reps` experimental repeats
# will be dumped in to.  Individual experimental settings / hyperparameters
# are set inside _launch_multiple.sh .  This is a bit messy, but there is
# practically no reason to change this file unless you want to change the
# number of experimental repeats or the `folder_head` variable.  Keeping
# track of, and making sure hyperparameters are consistent between the args
# python file, this file, the inner file is more hassle than it is worth.
# Therefore, don't touch this file too much and modify the inner file.

# IMPORTANT:
# This file must be called from the root directory.
# i.e.: ./cluster_scripts/launch_multiple.sh


# How many experimental repeats to run
reps=20


# Set the folder we will dump into.
folder_head="s4"                        # Change this to change the start of the folder name.
NOW="$(date '+%Y_%m_%d__%H_%M_%S')"     # Results folder will be time stamped.
folder="${folder_head}_${NOW}"          # Build folder string.


# Make sure there is a folder to dump SLURM reports in to.
mkdir -p "./Reports"


for(( seed=1; seed <= 10; seed++ )); do

    sbatch --export=ALL,SEED=$seed,FOLDER=$folder ./launch_scripts/_launch_multiple.sh

    # This is the line executed inside _launch_multiple.sh .
    # python3 ./tests/A2D/RunA2DExperiments.py --seed $seed --folder-extension folder --env-name 'MiniGrid-TigerDoorEnv-v0'
    
done
