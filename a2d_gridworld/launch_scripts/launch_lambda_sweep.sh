#!/bin/bash


# launch_lambda_sweep.sh
# Wrap launching the lambda gridworld experiments.
# This script will repeatedly call the SLURM job script _launch_lambda_sweep.sh
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
reps=10


# Set the folder we will dump into.
NOW="$(date '+%Y_%m_%d__%H_%M_%S')"
folder_head="lam_sweep"


# Make sure there is a folder to dump SLURM reports in to.
mkdir -p "./Reports"


lam=0.0
folder="${folder_head}_000_${NOW}"
for(( seed=1; seed <= $reps; seed++ )); do
    sbatch --export=ALL,LAM=$lam,SEED=$seed,FOLDER=$folder cluster_scripts/_launch_lambda_sweep.sh
done

lam=0.25
folder="${folder_head}_025_${NOW}"
for(( seed=1; seed <= $reps; seed++ )); do
    sbatch --export=ALL,LAM=$lam,SEED=$seed,FOLDER=$folder cluster_scripts/_launch_lambda_sweep.sh
done

lam=0.5
folder="${folder_head}_050_${NOW}"
for(( seed=1; seed <= $reps; seed++ )); do
    sbatch --export=ALL,LAM=$lam,SEED=$seed,FOLDER=$folder cluster_scripts/_launch_lambda_sweep.sh
done

lam=0.75
folder="${folder_head}_075_${NOW}"
for(( seed=1; seed <= $reps; seed++ )); do
    sbatch --export=ALL,LAM=$lam,SEED=$seed,FOLDER=$folder cluster_scripts/_launch_lambda_sweep.sh
done

lam=1.0
folder="${folder_head}_100_${NOW}"
for(( seed=1; seed <= $reps; seed++ )); do
    sbatch --export=ALL,LAM=$lam,SEED=$seed,FOLDER=$folder cluster_scripts/_launch_lambda_sweep.sh
done


 
