
# general imports
import argparse
import contextlib
import signal
import sys
import numpy as np
import os
import json
import torch
import matplotlib.pyplot as plt
import csv
from copy import deepcopy
import subprocess
import sys
import signal
import gym
import numpy
from pathlib import Path
from multiprocessing import Process, Manager
import time
import logging
import psutil
import multiprocessing
from multiprocessing import Process, Queue
import torch.multiprocessing as mp
import imageio
from torchvision.utils import save_image
import glob

# carla gym wrapper
from plai_carla import utils, gym_wrapper, displays
from plai_carla.utils import Resolution, Res

# inverted rl wrapper
from carla_a2d.evaluation.visualization import record_agent, generate_gif
from carla_a2d.evaluation.evaluate_agents import generate_eval_data
from carla_a2d.agent import Trainer as trainer

#
def create_agent_artifact(actor_critic_path, parser, file_name='video_ex.mp4', video_frames=1250):
    """
    Creates a set of Frontview, Birdview, and Third Person videos of both the network artifact
    as well as the expert policy defined by the environment.
    """
    # parse
    args = torch.load(actor_critic_path+'/args.pt')
    args.save_dir = actor_critic_path
    args.log_dir = actor_critic_path
    args.video = actor_critic_path
    # set cuda / device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.num_gpu = torch.cuda.device_count()
    args.headless = True
    args.log_dir = actor_critic_path+'/'
    args.video_length = video_frames

    #
    agent, agent_logging_info = evaluate_policy(args)

    # okkkey
    expert, expert_logging_info = evaluate_expert(args)

    # return tensor dict
    return agent_logging_info, expert_logging_info

# something to train the agent in carla
def evaluate_policy(params, verbose=True):
    """
    Evaluates the nueral network artifact trained using the set of scenarios
    specified. Only includes evaluation for the sampling distribution. This
    code also generats gifs and evaluation videos as well.
    """
    # initialize a trainer
    params.compact_state_filter_key = 0
    env_gen = lambda params_: gym.make(params.env_name, args=params)
    p = trainer.agent_from_params(params, env_gen=env_gen, add_train_environment=False)

    # reload info from dir
    p.load_model_checkpoint()

    # first evaluate the model
    print('evaluating...') if verbose else None
    with torch.no_grad():
        # now run an eval step
        if p.algo.actor_critic.sampling_dist == 'agent':
            eval_info = generate_eval_data(p, \
                max_evals = p.params.num_evals, type='agent', generate_gif=False)
        elif p.algo.actor_critic.sampling_dist == 'expert':
            eval_info = generate_eval_data(p, \
                max_evals = p.params.num_evals, type='expert', generate_gif=False)
    print(eval_info) if verbose else None
    print('evaluation complete...') if verbose else None

    # make a quick gif.
    print('generating model example video...')  if verbose else None
    with torch.no_grad():
        file_name='recording_ex.mp4'
        folder_name = 'gifs_set'
        record_agent(params, p.algo.actor_critic, file_name=file_name, max_time_steps=params.video_length)
        generate_gif(params, p.algo.actor_critic, folder_name=folder_name, max_time_steps=params.video_length)
    print('example video generated...') if verbose else None

    # return path to agent
    print('evaluation complete!')
    return p, eval_info

# something to train the agent in carla
def evaluate_expert(params, verbose=True):
    """
    Evaluates the expert policy which uses the set of scenarios
    specified. Only includes evaluation for the sampling distribution. This
    code also generats gifs and evaluation videos as well.
    """
    # initialize a trainer
    env_gen = lambda params_: gym.make(params.env_name, args=params)
    p = trainer.agent_from_params(params, env_gen=env_gen, add_train_environment=False)

    # set static expert policy
    p.algo.actor_critic.sampling_dist = 'expert'
    p.algo.actor_critic.use_learned_expert = False
    p.algo.actor_critic.beta = torch.tensor(1.)

    # first evaluate the model
    print('evaluating...') if verbose else None
    with torch.no_grad():
        eval_info = generate_eval_data(p, max_evals = p.params.num_evals, type='asymmetric_pid')
    print(eval_info) if verbose else None
    print('evaluation complete...') if verbose else None

    # make a quick gif.
    print('generating model example video...')  if verbose else None
    with torch.no_grad():
        file_name='expert_recording_ex.mp4'
        folder_name = 'expert_gifs_set'
        record_agent(params, p.algo.actor_critic, file_name=file_name, max_time_steps=params.video_length)
        generate_gif(params, p.algo.actor_critic, folder_name=folder_name, max_time_steps=params.video_length)
    print('example video generated...') if verbose else None

    # return path to agent
    print('evaluation complete!')
    return p, eval_info
