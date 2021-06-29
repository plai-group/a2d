#!/bin/bash
#SBATCH --job-name=s4                               # Set name of job.
#SBATCH --nodes=1 --ntasks=1                        # One task on one node (let PyTorch handle multiproc).
#SBATCH --cpus-per-task=4 --gpus=1                  # Set the number of CPUs and GPUs.
#SBATCH --time=23:59:00                             # Wallclock time.
#SBATCH --output=./Reports/s4_%A_%a.out             # Set text output (stdout) path.  Also echoed to log file.
#SBATCH --error=./Reports/s4_%A_%a.err              # Set error output (stderr) path.
#SBATCH --partition plai                            # Set the partition/queue on which to execute.

# _launch_multiple.sh
# This file is called repeatedly by `launch_multiple.sh`.  This was designed
# to make running the scripts as simple and transparent as possible on any
# system.  Individual hyperparameters / arguments can be set explicitly in
# here, although the defaults set to the values used for the experiments in
# the paper.

echo "Dumping into folder $FOLDER"
echo "Using seed $SEED"

python3 ./tests/A2D/RunA2DExperiments.py    --seed $SEED --folder-extension $FOLDER \
                                            --env-name 'MiniGrid-TigerDoorEnv-v0' \
                                            --rl-state              1 \
                                            --a2d-partial-state     1 \
                                            --ad-partial-observe    1 \
                                            --a2d-partial-observe   1 \
                                            --ete-partial-observe   1 \
                                            --arl-partial-observe   1 \
                                            --rl-partial-observe    1 \

