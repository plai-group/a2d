
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
from carla_a2d.memory.storage import RolloutStorage
from carla_a2d.utils.gen_utils import flatten_dict, generate_stats_dict
import plai_carla.utils as carla_utils
from carla_a2d.environments.envs import make_vec_envs
from carla_a2d.models.model import ActorCritic
import carla_a2d.utils.gen_utils as gen_utils

# run and evaluation cycle
def full_eval(self, max_evals = 25, act_type='', sampling_dist=None, verbose=False, use_running_avg=False):
    """
    Evaluates the performance of either the model or the expert examples depending on
    specified arguments.
    """
    # if sampling dist
    if sampling_dist is None:
        info_sampler = 0
        sampling_dist = self.algo.actor_critic.act
    elif sampling_dist == 'careless_actions':
        info_sampler = 'careless_actions'
    elif sampling_dist == 'careful_actions':
        info_sampler = 'careful_actions'
    else:
        raise Exception('provide valid sampler.')

    # check evaluation envs are set.
    if self.video_envs is None:
        self.params.compact_state_filter_key = 0
        eval_envs = make_vec_envs(self.params.env_name, self.params.seed, 1,
                        None, 'cuda:0', False, self.params)
        close_env = True
    else:
        eval_envs = self.video_envs
        close_env = False

    #
    eval_rollouts = RolloutStorage(1, 1, self.envs.observation_space,
        self.envs.action_space, self.algo.actor_critic.recurrent_hidden_state_size)

    # reset all environments
    obs = eval_envs.reset()
    for key in list(obs.keys()):
        eval_rollouts.obs[key][0].copy_(obs[key])
    eval_rollouts.to(self.device)

    # init some stuff
    episode_rewards, episode_lengths = [],[]
    additional_logging_info = []
    infos = None

    # now gather steps
    while len(episode_rewards) < max_evals:
        # Sample actions
        with torch.no_grad():
            if info_sampler:
                if infos is None:
                    action = eval_rollouts.obs['fixed_expert_actions'][0]
                elif info_sampler not in infos[0].keys():
                    action = eval_rollouts.obs['fixed_expert_actions'][0]
                elif info_sampler in infos[0].keys():
                    action = torch.tensor([infos[0][info_sampler]])
                else:
                    raise Exception()
            else:
                _, action, _, _ = sampling_dist({key:eval_rollouts.obs[key][0] for key in list(eval_rollouts.obs.keys())},device=self.device)
        # Observe reward and next obs
        obs, reward, done, infos = eval_envs.step(action)

        # new obs
        for key in list(obs.keys()):
            eval_rollouts.obs[key][0].copy_(obs[key])
        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        # get episode info
        for info in infos:
            if 'done' in info.keys():
                # rewards / lengths / info
                episode_rewards.append(info['done'][0])
                episode_lengths.append(info['done'][1])
                additional_logging_info.append(flatten_dict(info['additional_logs']))

    # convert logging stuff to dictionary
    reward_info = generate_stats_dict(act_type+'_eval_r',episode_rewards)
    length_info = generate_stats_dict(act_type+'_eval_h',episode_lengths)
    stacked_additional_info = {}
    for key in additional_logging_info[0].keys():
        vals = [additional_logging_info[i][key] for i in range(len(additional_logging_info))]
        stacked_additional_info.update(generate_stats_dict(act_type+'_eval_'+key, vals))

    if close_env:
        eval_envs.close()
        self.video_envs = None
    # return
    return {**reward_info, **length_info, **stacked_additional_info}

# create evaluation data
def generate_eval_data(self, max_evals, type, generate_gif=False):
    """
    Evaluates the performance of either the model or the expert examples depending on
    specified arguments. This function is an extension of the Trainer class.
    """
    # generate gifs
    if generate_gif:
        _, _, _, _, (gif_files_front, gif_files_bird) = generate_gif(actor_critic, 350,25)
        for idx, gif_file in enumerate(gif_files_front):
            wandb.log({"frontview_"+str(idx): wandb.Video(gif_file, fps=params.fps, format="gif")}, step=i)
        for idx, gif_file in enumerate(gif_files_bird):
            wandb.log({"birdview_"+str(idx): wandb.Video(gif_file, fps=params.fps, format="gif")}, step=i)
    # store initial values
    starting_dist = deepcopy(self.algo.actor_critic.sampling_dist)
    starting_beta = deepcopy(self.algo.actor_critic.beta)
    # evaluate birdview model
    if type == 'expert':
        self.algo.actor_critic.sampling_dist = 'expert'
        self.algo.actor_critic.beta = torch.tensor(0.)
        store_results = full_eval(self, max_evals = max_evals, act_type='expert')
    elif type == 'agent':
        # evaluate agent model
        self.algo.actor_critic.sampling_dist = 'agent'
        self.algo.actor_critic.beta = torch.tensor(0.)
        store_results = full_eval(self, max_evals = max_evals, act_type='agent')
    elif type == 'asymmetric_pid':
        # evaluate careless PID controller
        self.algo.actor_critic.beta = torch.tensor(1.)
        self.algo.actor_critic.use_learned_expert = 0
        store_results = full_eval(self, max_evals = max_evals, act_type='pid')
    elif type == 'careless_pid':
        store_results = full_eval(self, max_evals = max_evals, act_type='careless_pid',
            sampling_dist = 'careless_actions')
    elif type == 'careful_pid':
        store_results = full_eval(self, max_evals = max_evals, act_type='careful_pid',
            sampling_dist = 'careful_actions')
    # add in other info
    store_results['steps'] = self.steps
    store_results['time_steps'] = self.time_steps
    # reset
    self.algo.actor_critic.sampling_dist = starting_dist
    self.algo.actor_critic.beta = starting_beta
    # go through dict and detach everything
    for key in dict.keys(store_results):
        if torch.is_tensor(store_results[key]):
            store_results[key] = store_results[key].detach()
    # return set of eval valyes
    return store_results
