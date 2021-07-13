
# general imports
import os
import gym
import numpy as np
import torch
from gym.spaces.box import Box
import pybullet_envs
from copy import deepcopy

# baselines stuff
from carla_a2d.baselines.common.vec_env import VecEnvWrapper
from carla_a2d.baselines.common.vec_env.dummy_vec_env import ThreadVecEnv

# core lib stuff
from carla_a2d.environments.env_wrap import CarlaEnvWrapper

def make_env(env_id, seed, rank, log_dir, allow_early_resets, args):
    """
    Generator object for single vector environment. Returns a single envirnment.
    """
    def _thunk():
        # wrap this bad boi
        env = args.env_gen(None)
        env.seed(seed + rank)
        env = CarlaEnvWrapper(env, args, rank)
        return env
    return _thunk

def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  log_dir,
                  device,
                  allow_early_resets,
                  args,
                  num_frame_stack=None):
    """
    Generator object for threaded vector environment. Returns a single vectorized
    environment object.
    """
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, args)
        for i in range(num_processes)
    ]
    envs = ThreadVecEnv(envs)
    envs = VecPyTorch(envs, device)

    return envs

class VecPyTorch(VecEnvWrapper):
    """
    Vectorized Pytorch based environment
    """
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        for key in list(obs.keys()):
            obs[key] = torch.tensor(obs[key]).detach().to(self.device)
        return obs

    def step_async(self, actions):
        # Squeeze the dimension for discrete actions
        if isinstance(actions, torch.LongTensor):
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        for key in list(obs.keys()):
            obs[key] = torch.from_numpy(obs[key]).detach().to(self.device)
        # obs = torch.from_numpy(obs).detach().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
