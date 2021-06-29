#!/usr/bin/env bash

# launch_osc.sh
# Wrap launching the Q-function experiments.
# This script will repeatedly call the SLURM job script _launch_multiple_q.sh
# This outer script configures which environment to use and whether Q is being
# used or not.  Also sets the folder into which `reps` experimental repeats
# will be dumped in to.

# IMPORTANT:
# This file must be called from the root directory.
# i.e.: ./cluster_scripts/launch_multiple.sh


# How many experimental repeats to run
reps=10


# Set the folder we will dump into.
folder_head="q"                         # Change this to change the start of the folder name.
NOW="$(date '+%Y_%m_%d__%H_%M_%S')"     # Results folder will be time stamped.


# Make sure there is a folder to dump SLURM reports in to.
mkdir -p "./Reports"


# TIGER-DOOR-3, NO Q.
use_q=0
env='MiniGrid-TigerDoorEnvOSC-v3'
folder="${folder_head}_q${use_q}_${NOW}"
for(( seed=1; seed <= $reps; seed++ )); do
    sbatch --export=ALL,SEED=$seed,FOLDER=$folder,USE_Q=$use_q,ENV=$env cluster_scripts/_launch_multiple_osc.sh
done

# TIGER-DOOR-3, USE Q.
use_q=1
env='MiniGrid-TigerDoorEnvOSC-v3'
folder="${folder_head}_q${use_q}_${NOW}"
for(( seed=1; seed <= $reps; seed++ )); do
    sbatch --export=ALL,SEED=$seed,FOLDER=$folder,USE_Q=$use_q,ENV=$env cluster_scripts/_launch_multiple_osc.sh
done

# TIGER-DOOR-2, NO Q.
use_q=0
env='MiniGrid-TigerDoorEnvOSC-v2'
folder="${folder_head}_q${use_q}_${NOW}"
for(( seed=1; seed <= $reps; seed++ )); do
    sbatch --export=ALL,SEED=$seed,FOLDER=$folder,USE_Q=$use_q,ENV=$env cluster_scripts/_launch_multiple_osc.sh
done

# TIGER-DOOR-2, USE Q.
use_q=1
env='MiniGrid-TigerDoorEnvOSC-v2'
folder="${folder_head}_q${use_q}_${NOW}"
for(( seed=1; seed <= $reps; seed++ )); do
    sbatch --export=ALL,SEED=$seed,FOLDER=$folder,USE_Q=$use_q,ENV=$env cluster_scripts/_launch_multiple_osc.sh
done

# TIGER-DOOR-1, NO Q.
use_q=0
env='MiniGrid-TigerDoorEnvOSC-v1'
folder="${folder_head}_q${use_q}_${NOW}"
for(( seed=1; seed <= $reps; seed++ )); do
    sbatch --export=ALL,SEED=$seed,FOLDER=$folder,USE_Q=$use_q,ENV=$env cluster_scripts/_launch_multiple_osc.sh
done

# TIGER-DOOR-1, USE Q.
use_q=1
env='MiniGrid-TigerDoorEnvOSC-v1'
folder="${folder_head}_q${use_q}_${NOW}"
for(( seed=1; seed <= $reps; seed++ )); do
        sbatch --export=ALL,SEED=$seed,FOLDER=$folder,USE_Q=$use_q,ENV=$env cluster_scripts/_launch_multiple_osc.sh
done







