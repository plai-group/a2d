
# grab args
from get_args import get_args
import os
import psutil
import sys
from contextlib import redirect_stdout
import signal

# load in the trainer
from carla_a2d.trainer import train_agent
from carla_a2d.evaluator import create_agent_artifact
from carla_a2d.distillation.distil import distil_policy

# Enable non-graceful termination
def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

# Enable graceful termination
def sigterm_handler(signum, frame):
    raise InterruptedError("SIGTERM received")

# main function
def main():

    # set sig-kill handler
    signal.signal(signal.SIGTERM, sigterm_handler)

    # get our id
    proc_pid = os.getpid()

    # get rl args and the parser
    args, parser = get_args()

    # train agent via rl/ad/a2d
    if not args.skip_training:
        args.actor_critic_path = train_agent(parser)

    # generate a video of agent stored at path
    if args.generate_video:
        create_agent_artifact(args.actor_critic_path, parser)

    # distil preset list of models
    if args.model_distillation:
        args.actor_critic_path = distil_policy(parser)

    # kill em all
    kill(proc_pid)

# what to run when called
if __name__ == "__main__":
    main()
