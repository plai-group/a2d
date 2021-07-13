
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

# independent from primary class
def generate_gif(params, actor_critic, folder_name='gifs_folder', max_time_steps=350, max_evals = 5):

    """
    Creates a set of front view and bird view gifs using a specific actor
    critic nueral net artififact.
    """

    # push actor to gpu
    actor_critic.to("cuda:0", dtype=torch.float)

    # create some new envs
    params.compact_state_filter_key = 1
    envs = make_vec_envs(params.env_name, params.seed, 1,
                        None, 'cuda:0', False, params)

    # init storage
    memory = RolloutStorage(max_time_steps, 1,
                      envs.observation_space, envs.action_space,
                      actor_critic.recurrent_hidden_state_size)
    memory.to("cpu")


    # all gif files
    gif_files_front, gif_files_bird = [], []
    episode_rewards, episode_lengths = [],[]
    episode_waypoints_percent, episode_waypoints_hit = [],[]

    # will produce a video showing what's going on
    front_view_loc = params.log_dir+folder_name+'/front_view/'
    top_down_loc = params.log_dir+folder_name+'/bird_view/'
    gen_utils.cleanup_log_dir(front_view_loc)
    gen_utils.cleanup_log_dir(top_down_loc)

    try:
        os.mkdir(front_view_loc)
        os.mkdir(top_down_loc)
    except OSError:
        pass

    episode_rewards, episode_lengths = [],[]
    episode_waypoints_percent, episode_waypoints_hit = [],[]

    # logging info
    step, total_steps = 0, 0
    reset_step = False
    files_1=[]
    files_2=[]
    evals = 0

    # initialize environment
    obs = envs.reset()
    for key in list(obs.keys()):
        memory.obs[key][0].copy_(obs[key])

    # now gather steps
    while evals < max_evals:

        # convert states
        for key in obs.keys():
            obs[key] = obs[key].float().to("cuda:0")
        # generate action
        value, action, action_log_prob, _ = actor_critic.act(obs, deterministic=False, device="cuda:0")
        # step environemnt
        next_obs, reward, done, infos = envs.step(action.detach())
        # If done then clean the history of observations.
        next_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        next_bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
        # add everything to roll-outs
        memory.insert(next_obs, memory.recurrent_hidden_states[0], action, action_log_prob, reward, reward, next_masks, next_bad_masks)
        obs = deepcopy(next_obs)
        memory.to('cpu')

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

        # get episode info
        for info in infos:
            if 'done' in info.keys():
                # rewards / lengths info
                episode_rewards.append(info['done'][0])
                episode_lengths.append(info['done'][1])
                # waypoints info
                episode_waypoints_percent.append(info['waypoints_info'][0]/info['waypoints_info'][1])
                episode_waypoints_hit.append(info['waypoints_info'][0])
                # reset flag
                reset_step = True
                # increment the number of evaluations
                evals += 1

        # if the ep is completed or we reached the max gif steps
        if (step == max_time_steps) or reset_step:
            # now generate the gif
            if (len(files_1) > 1) and (len(files_2) > 1):
                with imageio.get_writer(params.log_dir+folder_name+'/'+str(evals)+'_front_view.gif', mode='I',duration=0.2) as writer:
                    for filename in files_1:
                        image = imageio.imread(filename)
                        writer.append_data(image)
                gif_files_front.append(params.log_dir+folder_name+'/'+str(evals)+'_front_view.gif')
                with imageio.get_writer(params.log_dir+folder_name+'/'+str(evals)+'_bird_view.gif', mode='I',duration=0.2) as writer:
                    for filename in files_2:
                        image = imageio.imread(filename)
                        writer.append_data(image)
                gif_files_bird.append(params.log_dir+folder_name+'/'+str(evals)+'_bird_view.gif')

            # set new files
            files_1=[]
            files_2=[]

            # delete files in these folders
            for f in glob.glob(front_view_loc+'*'):
                os.remove(f)
            for f in glob.glob(top_down_loc+'*'):
                os.remove(f)

            # reset the step if we have concluded the ep
            if reset_step:
                reset_step = False
                step = 0
            else:
                step += 1

        # keep running withhout storing gifs if we have not ended
        elif step > max_time_steps:
            # if we have hit a reset finally
            if reset_step:
                reset_step = False
                step = 0
            else:
                step += 1

        # or continue to save gifs
        elif step < max_time_steps:
            # save images
            save_image(obs['frames'][0,0,...].detach(), front_view_loc+'front'+str(step)+'.png')
            save_image(obs['birdview'][0,0,...].detach(), top_down_loc+'topdown'+str(step)+'.png')
            # store file names
            files_1.append(front_view_loc+'front'+str(step)+'.png')
            files_2.append(top_down_loc+'topdown'+str(step)+'.png')
            # update step
            step += 1

    # convert logging stuff to dictionary
    reward_info = {}
    reward_info = generate_stats_dict('eval_r',episode_rewards)
    length_info = generate_stats_dict('eval_h',episode_lengths)
    wpp_info = generate_stats_dict('eval_wpp',episode_waypoints_percent)
    wph_info = generate_stats_dict('eval_wph',episode_waypoints_hit)

    # remove redunant dir
    os.rmdir(front_view_loc)
    os.rmdir(top_down_loc)

    # close environment
    envs.close()
    envs = None
    # return
    return reward_info, length_info, wpp_info, wph_info, (gif_files_front, gif_files_bird)

# generates a video
def record_agent(params, actor_critic, file_name='video_ex.mp4', max_time_steps=1250, early_stop=True):
    """
    Creates third person mp4 video of Actor Critic nueral net artifact. This
    video is higher quality and includes color.
    """
    # push actor to gpu
    actor_critic.to("cuda:0", dtype=torch.float)
    # create some new envs
    params.compact_state_filter_key = 0
    envs = make_vec_envs(params.env_name, params.seed, 1,
                        None, 'cuda:0', False, params)
    # set up rendering
    env = envs.envs[0]
    env.env.render()
    params.video = params.log_dir+file_name
    if params.video is not None:
        env.env.render(mode='video', filename=params.video, frames=max_time_steps)
    # init storage
    memory = RolloutStorage(max_time_steps, 1,
                      envs.observation_space, envs.action_space,
                      actor_critic.recurrent_hidden_state_size)
    memory.to("cpu")
    logging_info = []
    # initialize environment
    obs = envs.reset()
    for key in list(obs.keys()):
        memory.obs[key][0].copy_(obs[key])
    # set up rendering
    for step in range(max_time_steps):
        # convert states
        for key in obs.keys():
            obs[key] = obs[key].float().to("cuda:0")
        # generate action
        value, action, action_log_prob, _ = actor_critic.act(obs, deterministic=False, device="cuda:0")
        # step environemnt
        next_obs, reward, done, infos = envs.step(action.detach())
        # If done then clean the history of observations.
        next_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        next_bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
        # add everything to roll-outs
        memory.insert(next_obs, memory.recurrent_hidden_states[0], action, action_log_prob, reward, reward, next_masks, next_bad_masks, k=step)
        obs = deepcopy(next_obs)
        memory.to('cpu')
        # logging
        for info in infos:
            if 'done' in info.keys():
                logging_info.append(sum([info['additional_logs']['episode_waypoints_percent'] for i in info]) / len([info['additional_logs']['episode_waypoints_percent'] for i in info]))
                # early stop
                if early_stop:
                    # close environment
                    envs.close()
                    envs = None
                    # return it
                    return logging_info
    # close environment
    envs.close()
    envs = None
    # return it
    return logging_info
