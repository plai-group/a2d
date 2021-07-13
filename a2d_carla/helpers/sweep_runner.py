# general imports
import argparse
import sys
import os
from copy import deepcopy
import subprocess
from pathlib import Path
import time

def eval_generation(job_name='1', machine='borg', command='', directory='.'):
    # make thhe directory if it does not exist
    Path('eval_dir').mkdir(parents=True, exist_ok=True)
    # set the base runner
    args_list = sys.argv[1:]
    if machine=='borg':
        file = open('eval_dir/job_'+job_name+'.sh',"w+")
        file.write('#!/bin/sh \n')
        file.write('#SBATCH --partition=plai \n')
        file.write('#SBATCH --gres=gpu:4 \n')
        file.write('#SBATCH --cpus-per-gpu=5 \n')
        file.write('#SBATCH --nodes=1 \n')
        file.write('#SBATCH --time=00-24:00     # time (DD-HH:MM) \n')
        file.write('cd ' + directory + ' \n')
        file.write(command + ' \n')
        file.write('exit')
        file.close()
    elif machine=='borg-long':
        file = open('eval_dir/job_'+job_name+'.sh',"w+")
        file.write('#!/bin/sh \n')
        file.write('#SBATCH --partition=plai \n')
        file.write('#SBATCH --gres=gpu:4 \n')
        file.write('#SBATCH --cpus-per-gpu=5 \n')
        file.write('#SBATCH --nodes=1 \n')
        file.write('#SBATCH --time=00-36:00     # time (DD-HH:MM) \n')
        file.write('cd ' + directory + ' \n')
        file.write(command + ' \n')
        file.write('exit')
        file.close()
    elif machine=='borg-small':
        file = open('eval_dir/job_'+job_name+'.sh',"w+")
        file.write('#!/bin/sh \n')
        file.write('#SBATCH --partition=plai \n')
        file.write('#SBATCH --gres=gpu:1 \n')
        file.write('#SBATCH --cpus-per-gpu=5 \n')
        file.write('#SBATCH --time=00-24:00     # time (DD-HH:MM) \n')
        file.write('cd ' + directory + ' \n')
        file.write(command + ' \n')
        file.write('exit')
        file.close()
    elif machine=='cedar-long':
        file = open('eval_dir/job_'+job_name+'.sh',"w+")
        file.write('#!/bin/sh \n')
        file.write('#SBATCH --account=rrg-schmidtm \n')
        file.write('#SBATCH --mem=12G \n')
        file.write('#SBATCH --cpus-per-task=8 \n')
        file.write('#SBATCH --time=00-08:00     # time (DD-HH:MM) \n')
        file.write('#SBATCH --gres=gpu:p100:1 \n')
        file.write('cd ' + directory + ' \n')
        file.write('export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \n')
        file.write(command + ' \n')
        file.write('exit')
        file.close()
    else:
        file = open('eval_dir/job_'+job_name+'.sh',"w+")
        file.write('#!/bin/sh \n')
        file.write('#SBATCH --account=rrg-schmidtm \n')
        file.write('#SBATCH --mem=12G \n')
        file.write('#SBATCH --cpus-per-task=8 \n')
        file.write('#SBATCH --time=00-06:00     # time (DD-HH:MM) \n')
        file.write('#SBATCH --gres=gpu:p100:1 \n')
        file.write('cd ' + directory + ' \n')
        file.write('export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \n')
        file.write(command + ' \n')
        file.write('exit')
        file.close()
    # create the command
    command = 'sbatch ' + 'eval_dir/job_'+job_name+'.sh'
    # # submit the job
    exit_status = subprocess.call(command, shell=True)
    exit_status = subprocess.call('echo "\nsubmitted:"', shell=True)
    command = 'cat eval_dir/job_'+job_name+'.sh'
    exit_status = subprocess.call(command, shell=True)
    exit_status = subprocess.call('echo "\n"', shell=True)
    # now delete the file
    os.remove('eval_dir/job_'+job_name+'.sh')
    os.rmdir('eval_dir')
    # nothing to return
    return None

def main():
    parser = argparse.ArgumentParser(description='job runner')
    parser.add_argument('--directory', default='..')
    parser.add_argument('--command', default='echo "Im afraid I cant do that, Dave."')
    parser.add_argument('--machine', default='borg')
    parser.add_argument('--job_name', default='1')
    args = parser.parse_args()
    eval_generation(machine=args.machine, command=args.command, directory=args.directory, job_name=args.job_name)

if __name__ == "__main__":
    main()
