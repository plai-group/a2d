# general imports
import numpy as np
import os
import torch
import gym
import numpy
from pathlib import Path
from os import path
from collections import deque
import wandb

# carla gym wrapper
from plai_carla.utils import Resolution, Res
import plai_carla.utils as carla_utils
from plai_carla import utils, gym_wrapper, displays

#
from carla_a2d.evaluation.visualization import record_agent, generate_gif
from carla_a2d.evaluation.evaluate_agents import generate_eval_data, full_eval

# plai rl lib imports
from carla_a2d.agent import Trainer as trainer

# primary training loop
def run_trainer(p, trainer_steps, eval_info):
    """
    Single training loop for RL/A2D/AD
    """
    closed = False
    for i in range(trainer_steps):
        # now the main update loop
        try:
            p.train_step()
        # close env to avoid eof
        except KeyboardInterrupt:
            p.envs.close()
            if p.video_envs is not None:
                p.video_envs.close()
            break
    return p, eval_info, closed

def init_trainer(parser):
    """
    Initializes some parameter values required to run specific environments.
    """
    # need to re-parse
    carla_utils.add_driving_options(parser)
    gym_wrapper.CarlaEnv.add_command_line_options(parser)

    # re-parse
    args = parser.parse_args()
    if args.force_fps is None:
        args.fps = 10
    else:
        args.fps = args.force_fps

    # update args
    args.frontview_res = Res.CIL
    args.birdview_res = Res.BIRDVIEW
    args.expert_res = Res.BIRDVIEW
    args.agent_res = Res.CIL

    # set cuda / device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.num_gpu = torch.cuda.device_count()
    args.headless = True

    return args

def eval_trainer(p, model_evals, wpp_evals, eval_info):
    """
    Evaluation wrapper which calls lower level functions.
    """
    # now run an eval step
    if p.algo.actor_critic.sampling_dist == 'agent':
        eval_info = generate_eval_data(p, \
            max_evals = p.params.num_evals, type='agent', generate_gif=False)
    elif p.algo.actor_critic.sampling_dist == 'expert':
        eval_info = generate_eval_data(p, \
            max_evals = p.params.num_evals, type='expert', generate_gif=False)
    else:
        raise Exception()

    # log evaluation info
    model_evals.append(eval_info[p.algo.actor_critic.sampling_dist+'_eval_r_mean'])
    wpp_evals.append(eval_info[p.algo.actor_critic.sampling_dist+'_eval_episode_waypoints_percent_mean'])
    eval_info[p.algo.actor_critic.sampling_dist+'_avg_eval_r_mean'] = sum(model_evals) / len(model_evals)
    eval_info[p.algo.actor_critic.sampling_dist+'_avg_eval_episode_waypoints_percent_mean'] = sum(wpp_evals) / len(wpp_evals)

    # log the eval info with wandb
    wandb.log(eval_info, step=p.steps)

    # return stuff
    return eval_info, wpp_evals, model_evals

# something to train the agent in carla
def train_agent(parser, verbose=False):
    """
    Primary training loop which calls lower level functions.
    """
    # grab initialied parameters
    params = init_trainer(parser)

    # init wandb + modify saved model info
    run = wandb.init(project=params.project, group=params.group, config=vars(params))
    params.save_dir = params.save_dir + params.env_name + '/' + wandb.run.id + '/'
    params.log_dir = params.log_dir + params.env_name + '/' + wandb.run.id + '/'

    # initialize a trainer
    env_gen = lambda params_: gym.make(params.env_name, args=params)
    p = trainer.agent_from_params(params, env_gen=env_gen)
    torch.save(parser.parse_args(), params.save_dir + 'args.pt')
    eval_info = None

    # now call step
    closed, i, model_evals, wpp_evals = False, 0,  deque([],maxlen=25), deque([],maxlen=25)

    # determine if we will eval
    if p.params.eval_interval > 0:
        include_eval_step = True
        train_steps = 1 + int(p.params.eval_interval/(params.num_steps * params.num_processes))
    else:
        include_eval_step = False
        train_steps = 1

    # first evaluate the model
    print('evaluating...') if verbose else None
    with torch.no_grad():
        p.model_checkpoint()
        eval_info, wpp_evals, model_evals = eval_trainer(p, model_evals, wpp_evals, eval_info)
    print('evaluation complete...') if verbose else None

    # now iterate
    while i < p.num_updates:

        # create intermediate examples for demonstration
        if params.save_intermediate_video:
            print('generating model example video...')  if verbose else None
            with torch.no_grad():
                file_name='recording_ex_'+str(i)+'.mp4'
                folder_name = 'gifs_set_'+str(i)
                record_agent(params, p.algo.actor_critic, file_name=file_name, max_time_steps=1250)
                generate_gif(params, p.algo.actor_critic, folder_name=folder_name, max_time_steps=1250)
            print('example video generated...') if verbose else None

        # run a set of updates
        print('taking a training step...') if verbose else None
        run_trainer_args = (p, train_steps, eval_info)
        p, iter_info, closed = run_trainer(*run_trainer_args)
        i += train_steps
        print('completed a training step...') if verbose else None

        # now evaluate the model
        print('evaluating...') if verbose else None
        with torch.no_grad():
            p.model_checkpoint()
            eval_info, wpp_evals, model_evals = eval_trainer(p, model_evals, wpp_evals, eval_info)
        print('evaluation complete...') if verbose else None

    # close env to avoid eof
    print('closing simulators...') if verbose else None
    if not closed:
        p.envs.close()
        if p.video_envs is not None:
            p.video_envs.close()
    print('simulators closed...') if verbose else None

    # make a longer gif.
    print('generating model example video...')  if verbose else None
    with torch.no_grad():
        file_name='recording_ex_final.mp4'
        folder_name = 'gifs_set_final'
        p.model_checkpoint()
        record_agent(params, p.algo.actor_critic, file_name=file_name, max_time_steps=1250)
        generate_gif(params, p.algo.actor_critic, folder_name=folder_name, max_time_steps=1250)
    print('example video generated...') if verbose else None

    # return path to agent
    print('training complete!')
    return p
