#!/bin/bash
#SBATCH --job-name=q                                # Set name of job.
#SBATCH --nodes=1 --ntasks=1                        # One task on one node (let PyTorch handle multiproc).
#SBATCH --cpus-per-task=4 --gpus 1                  # Set the number of CPUs and GPUs.
#SBATCH --time=23:59:00                             # Wallclock time.
#SBATCH --output=./Reports/q_%A_%a.out              # Set text output (stdout) path.  Also echoed to log file.
#SBATCH --error=./Reports/q_%A_%a.err               # Set error output (stderr) path.
#SBATCH --partition plai                            # Set the partition/queue on which to execute.

echo "Dumping into folder $FOLDER"
echo "Using seed $SEED"

python3 ./tests/A2D/RunA2DExperiments.py --seed $SEED --folder-extension $FOLDER --env-name $ENV --USE-Q $USE_Q
