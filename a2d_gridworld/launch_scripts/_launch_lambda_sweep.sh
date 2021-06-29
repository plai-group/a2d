#!/bin/bash
#SBATCH --job-name=lam_sweep                        # Set name of job.
#SBATCH --nodes=1 --ntasks=1                        # One task on one node (let PyTorch handle multiproc).
#SBATCH --cpus-per-task=4 --gpus=1                  # Set the number of CPUs and GPUs.
#SBATCH --time=47:59:00                             # Wallclock time.
#SBATCH --output=./Reports/lam_sweep_%A_%a.out      # Set text output (stdout) path.  Also echoed to log file.
#SBATCH --error=./Reports/lam_sweep_%A_%a.err       # Set error output (stderr) path.
#SBATCH --partition plai                            # Set the partition/queue on which to execute.

# _launch_multiple.sh
# This file is called repeatedly by `launch_multiple.sh`.  This was designed
# to make running the scripts as simple and transparent as possible on any
# system.  Individual hyperparameters / arguments can be set explicitly in
# here, although the defaults set to the values used for the experiments in
# the paper.  The main different here is that lambda value used for RL and
# A2D are explicitly set.

echo "Dumping into folder $FOLDER"
echo "Using seed $SEED"

xvfb-run-safe python3 ./tests/A2D/RunA2DExperiments.py  --seed $SEED --folder-extension $FOLDER \
                                                        --lambda-rl $LAM --lambda-a2d $LAM \
                                                        --env-name 'MiniGrid-TigerDoorEnvOSC-v2' --USE-Q 0
