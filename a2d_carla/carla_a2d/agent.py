
# general imports
import copy
import glob
import os
import time
from collections import deque
from statistics import mean, stdev
import psutil
import datetime
import gc
import random
import collections

# deep learning imports
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader

# load in library stuff
from carla_a2d.algorithms.a2d_class import A2D
from carla_a2d.algorithms.ad_class import AD
from carla_a2d.algorithms.rl_class import RL
from carla_a2d.algorithms.bc_class import BC
from carla_a2d.models.model import ActorCritic
import carla_a2d.utils.gen_utils as utils
import carla_a2d.utils.torch_utils as torch_utils
import carla_a2d.utils.optim_utils as optim_utils
from carla_a2d.environments.envs import make_vec_envs
from carla_a2d.memory.storage import RolloutStorage

# logging
import wandb

# main trainer class
class Trainer():
    """
    This is the core training file for A2D, PPO, and AD methods used in the
    'Robust asymmetric learning for POMDPs' Paper found here:
    https://arxiv.org/abs/2012.15566. It is broken up into training steps, both
    RL based, as well as IL based, online evaluation, logging, and display. It
    also includes methods to sample from the environment, and store data. All
    static methods are included for the startup of the algorithm block, and
    assign local variables based on the algorithm being learned like beta.

    Attributes
    ----------
    actor_critic_class: nn.Module
    Actor critic module used to learn a control policy within CARLA. This must
    be able to sample, evaluate and compute things like log probability of actions,
    value functions, and internal policy divergences.

    params: Namespace
    List of parameters provided by user arguments.

    env_device: str
    Device which to store the set of environments on. This currently assumes a
    single device, and does not destribute them in a particularly intelligent
    way. Future iterations may address this.

    add_eval_environment: bool
    Whether to include an environemnt dedicated specifically to evaluation, or
    if the experiments will ignore this step due to computational constraints.

    Methods
    -------
    sample()
    Samples a set of examples from the environment, and stores them in the
    current replay buffer construct. Also logs all relevent statistics desired
    by the user via averaging.

    eval()
    Returned logged statistics from the current set of environment interactions

    IL_step()
    Projects the network associated with the sampling distribution onto either
    a fixed expert distribution or the differentiable expert being learning
    via the A2D update online.

    RL_step()
    Updates either the expert or agent policy via RL depending on the algorithm,
    as well as whatever value function update is required.

    train_step()
    Handles a single training step for RL in the POMDP/MDP, AD in the POMDP/MDP,
    or A2D in both the POMDP and MDP

    model_checkpoint()
    Saves the current multi-optimizer and modelparameters for restarts.

    log()
    Logs information about training proccess to csv file, as well as wandb
    when this option is selected.

    display()
    prints desired information about the training proccess.

    Static-Methods
    -------
    model_init(actor_critic, params)
    Sets important parameters in actor critic model based on algorithm and
    the specified user parameters.

    load_checkpoint(actor_critic, params)
    reload checkpoint which includes model, optimizer, and various statistics.

    agent_from_params(params,env_gen,ac_class,add_eval_environment)
    Creates the agent class from the stored parameters.
    """
    def __init__(self, actor_critic_class, params, env_gen=None, env_device="cuda:1",
                    add_train_environment=True):

        # set lr
        params.lr = 10**params.log_lr
        self.lr = params.lr

        # set state-filtering key
        params.compact_state_filter_key = -1

        # get parameters
        self.params = params
        self.params.env_device = env_device
        # add this just in case we have a pre-built
        self.params.env_gen = env_gen
        self.logging_map = None

        # clean and set log dir
        utils.cleanup_log_dir(self.params.log_dir)
        utils.cleanup_log_dir(self.params.save_dir)

        # create direct place to store results manually
        self.results_path = os.path.join(self.params.log_dir, 'results.csv')

        # set device info
        self.env_device = torch.device(utils.device_mask(env_device, params.num_gpu) if params.cuda else "cpu")
        self.device = torch.device(utils.device_mask("cuda:1", params.num_gpu) if params.cuda else "cpu")

        # set environment info
        if add_train_environment:
            self.add_train_environment = add_train_environment
            self.envs = make_vec_envs(params.env_name, params.seed, params.num_processes,
                            None, self.env_device, False, self.params)
        else:
            self.add_train_environment = add_train_environment
            self.envs = make_vec_envs(params.env_name, params.seed, 1,
                            None, self.env_device, False, self.params)

        # check if we want an outer env
        self.video_envs = None

        # set actor / critic policy info
        actor_critic = actor_critic_class(self.envs.observation_space,
            self.envs.action_space, base_kwargs={'recurrent': False, 'params': params})

        # set  RL algorithm
        if self.params.algo_key in ['rl-agent', 'rl-expert']:
            self.algo = RL(actor_critic, params=self.params, device=self.device)
        elif self.params.algo_key in ['ad-agent', 'ad-expert']:
            self.algo = AD(actor_critic, params=self.params, device=self.device)
        elif self.params.algo_key in ['a2d-agent']:
            self.algo = A2D(actor_critic, params=self.params, device=self.device)
        elif self.params.algo_key in ['bc-agent', 'bc-expert']:
            self.algo = BC(actor_critic, params=self.params, device=self.device)
        else:
            raise NotImplementedError

        # some init arguments
        self.algo.actor_critic = self.model_init(self.algo.actor_critic, self.params)

        # move network and rollouts to device 1
        self.algo.actor_critic = self.algo.actor_critic.to(self.device)

        # set roll out info / storage
        self.rollouts = RolloutStorage(params.num_steps, params.num_processes,
                          self.envs.observation_space, self.envs.action_space,
                          self.algo.actor_critic.recurrent_hidden_state_size)

        # intialize everything
        obs = self.envs.reset()
        for key in list(obs.keys()):
            self.rollouts.obs[key][0].copy_(obs[key])
        self.rollouts.to(self.device)

        #  log + cpi
        self.episode_rewards = deque(maxlen=params.logged_moving_avg)
        self.lagged_episode_rewards = deque(maxlen=2*params.logged_moving_avg)

        # log + cpi
        self.value_loss = deque(maxlen=params.logged_moving_avg)
        self.lagged_value_loss = deque(maxlen=2*params.logged_moving_avg)

        # other logging info
        self.episode_lengths = deque(maxlen=params.logged_moving_avg)
        self.execution_time = deque(maxlen=10*params.logged_moving_avg)
        self.additional_logging_info = deque(maxlen=params.logged_moving_avg)

        # set init steps
        self.steps, self.time_steps = 0, 0
        self.num_updates = int(params.train_steps) if params.num_env_steps < 0 \
            else int(params.num_env_steps/(params.num_steps * params.num_processes))

        # move the agent to the sim device
        self.algo.actor_critic.to(self.device)

        # close environments if we dont want to train
        if not add_train_environment:
            self.envs.close()

        # create some new envs
        self.params.compact_state_filter_key = 0

    def sample(self, sampling_dist=None, verbose=False):

        # episode rewards
        my_ep_reward_logger = 0

        # move the agent to the sim device
        self.algo.actor_critic.to(self.device)

        # now gather steps
        for step in range(self.params.num_steps):
            # Sample actions
            with torch.no_grad():

                if sampling_dist is None:
                    value, action, action_log_prob, recurrent_hidden_states = self.algo.actor_critic.act(
                        {key:self.rollouts.obs[key][step] for key in list(self.rollouts.obs.keys())},
                        rnn_hxs=self.rollouts.recurrent_hidden_states[step],
                        masks=self.rollouts.masks[step], device=self.device)
                else:
                    value, action, action_log_prob, recurrent_hidden_states = sampling_dist(
                        {key:self.rollouts.obs[key][step] for key in list(self.rollouts.obs.keys())},
                        rnn_hxs=self.rollouts.recurrent_hidden_states[step],
                        masks=self.rollouts.masks[step], device=self.device)
            # Observe reward and next obs
            start = time.time()
            next_obs, reward, done, infos = self.envs.step(action)

            end = time.time()
            self.execution_time.append(end-start)

            # get episode info
            for info in infos:
                if 'done' in info.keys():
                    # rewards / lengths info
                    self.episode_rewards.append(info['done'][0])
                    self.lagged_episode_rewards.append(info['done'][0])
                    assert print(info['done'][0]) if not info['done'][0] == info['done'][0] else 1
                    self.episode_lengths.append(info['done'][1])
                    self.additional_logging_info.append(utils.flatten_dict(info['additional_logs']))

            # If done then clean the history of observations.
            next_masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            next_bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            # fix with the end of episode observations
            try:
                obs['invasion_ind'] = torch.FloatTensor(
                    [[1.*info['invasion']] for info in infos])
                obs['crash_ind'] = torch.FloatTensor(
                    [[1.*info['collision']] for info in infos])
            except:
                pass
            # store reward as value
            if value == None:
                value = reward
            # add everything to roll-outs
            self.rollouts.insert(next_obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, next_masks, next_bad_masks, k=step)
        # increment time-steps
        self.time_steps += self.params.num_processes * self.params.num_steps

        # compute returns
        with torch.no_grad():
            # compute expected next value
            next_value = self.algo.actor_critic.get_value(
                {key:self.rollouts.obs[key][-1] for key in list(self.rollouts.obs.keys())},
                rnn_hxs=self.rollouts.recurrent_hidden_states[-1],
                masks=self.rollouts.masks[-1], device=self.device).detach()
            # mask values where the game ends
            next_value = next_value*self.rollouts.masks[-1]
            # update rollouts
            self.rollouts.compute_returns(next_value, self.params.use_gae, self.params.gamma,
                                 self.params.gae_lambda, self.params.use_proper_time_limits)

        # move the agent + rollouts back to device them back
        self.rollouts.to(self.device)

        # returns
        return self.rollouts

    def eval(self):

        # create dicts
        if len(self.episode_rewards) < 2:
            reward_info = utils.generate_stats_dict('r',[],if_empty_eval=-np.Inf)
            length_info = utils.generate_stats_dict('h',[],if_empty_eval=0)
            stacked_additional_info = {}
        else:
            reward_info = utils.generate_stats_dict('r',self.episode_rewards)
            length_info = utils.generate_stats_dict('h', self.episode_lengths)
            stacked_additional_info = {}
            for key in self.additional_logging_info[0].keys():
                vals = [self.additional_logging_info[i][key] for i in range(len(self.additional_logging_info))]
                stacked_additional_info.update(utils.generate_stats_dict(key, vals))

        # also add beta as a metric
        stacked_additional_info['beta'] = self.algo.actor_critic.beta
        stacked_additional_info['steps'] = self.steps
        stacked_additional_info['time_steps'] = self.time_steps

        # return
        return {**reward_info, **length_info, **stacked_additional_info}

    def IL_step(self, sample, verbose=False):

        # update
        loss_info= self.algo.project(sample)

        return loss_info

    def RL_step(self, rollouts, verbose=False):

        # if we include a policy delay (TD3 style)
        policy_update = True
        if self.params.delayed_policy_update > 0:
            if (self.steps + 1) % self.params.delayed_policy_update != 0:
                policy_update = False

        # initial pre-training of critic
        if self.params.pretrain_critic_updates > self.steps + 1:
            policy_update = False

        # multi updates
        for _ in range(self.params.RL_updates_per_batch):
            loss_info = self.algo.improve(rollouts, policy_update=policy_update)

        # nothing to return
        return loss_info

    def train_step(self, verbose=False):

        # small inits
        il_info, rl_info = {}, {}

        # interact with environment
        with torch.no_grad():
            print('sampling ....', datetime.datetime.now()) if verbose else None
            saps = self.sample()
            print('sampled.', datetime.datetime.now()) if verbose else None

        # take improvement steps
        if self.params.algo_key in ['rl-expert', 'rl-agent', 'a2d-agent']:
            print('RL step ....', datetime.datetime.now()) if verbose else None
            rl_info = self.RL_step(saps)
            print('RL step complete.', datetime.datetime.now()) if verbose else None

        # take projection steps
        if self.params.algo_key in ['ad-expert', 'ad-agent', 'a2d-agent']:
            print('DAgger step ....', datetime.datetime.now()) if verbose else None
            il_info = self.IL_step(saps)
            print('DAgger step complete.', datetime.datetime.now()) if verbose else None

        # evaluation/logging/display
        with torch.no_grad():

            print('increment info ...', self.steps+1) if verbose else None
            self.steps += 1
            self.algo.actor_critic.beta *= self.params.beta_update
            self.rollouts.after_update()
            print('increment info ...', self.steps+1) if verbose else None

            print('update lr ...', self.steps+1) if verbose else None
            if self.params.use_linear_lr_decay:
                self.lr, self.algo.optimizer = optim_utils.update_linear_schedule(self.algo.optimizer, self.steps,
                    self.num_updates, self.params.lr, multiple_opt=True)
            print('new lr / beta:', (self.lr,self.algo.actor_critic.beta)) if verbose else None

            # logging/display/eval
            if self.steps % self.params.log_interval == 0:

                print('evaluating...', datetime.datetime.now()) if verbose else None
                eval_info = self.eval()
                eval_info.update(rl_info)
                eval_info.update(il_info)
                print('rl loss info:', rl_info) if verbose else None
                print('il loss info:', il_info) if verbose else None
                print('evaluated.', datetime.datetime.now()) if verbose else None

                print('displaying...', datetime.datetime.now()) if verbose else None
                self.display(eval_info)
                print('displayed...', datetime.datetime.now()) if verbose else None

                print('logging.', datetime.datetime.now()) if verbose else None
                self.log(eval_info)
                print('logged.', datetime.datetime.now()) if verbose else None

            else:
                print('no logging/display/eval...', datetime.datetime.now()) if verbose else None
                eval_info = rl_info
                print('moving to next iter...', datetime.datetime.now()) if verbose else None

            #
            if self.steps % self.params.save_interval == 0:
                print('saving model.', datetime.datetime.now()) if verbose else None
                self.model_checkpoint()
                print('model and optimizer saved.', datetime.datetime.now()) if verbose else None

            # return
            return saps, eval_info

    def model_checkpoint(self):
        # save network info
        torch.save(self.algo.actor_critic.state_dict(), self.params.save_dir+"/model.pt")
        # save optimizer info
        torch.save(self.algo.optimizer.state_dict(), self.params.save_dir+"/optimizer.pt")
        # save state normalization
        torch.save([env.compact_state_filter for env in self.envs.unwrapped.envs], \
            self.params.save_dir+'/state_filters.pt')
        # save a dictionary with restart-info
        restart_dict = {'steps': self.steps, 'time_steps':self.time_steps,
                        'episode_rewards':self.episode_rewards,
                        'episode_lengths':self.episode_lengths,
                        'additional_info':self.additional_logging_info,
                        'execution_time': self.execution_time,
                        'current_lr': self.lr}
        torch.save(restart_dict, self.params.log_dir+"/restart_dict.pt")

    def load_model_checkpoint(self):
        # save network info
        self.algo.actor_critic.load_state_dict(torch.load(self.params.save_dir+"/model.pt"))
        self.algo.optimizer.load_state_dict(torch.load(self.params.save_dir+"/optimizer.pt"))
        # close current envs and reload stored ones
        if self.add_train_environment:
            self.envs.close()
            self.params.compact_state_filter_key = 0
            self.envs = make_vec_envs(self.params.env_name, self.params.seed, self.params.num_processes,
                        None, self.env_device, False, self.params)
        restart_dict = torch.load(self.params.log_dir+"/restart_dict.pt")
        # reload other incremental info
        self.steps = restart_dict['steps']
        self.time_steps = restart_dict['time_steps']
        self.episode_rewards = restart_dict['episode_rewards']
        self.episode_lengths = restart_dict['episode_lengths']
        self.additional_logging_info = restart_dict['additional_info']
        self.execution_time = restart_dict['execution_time']
        self.lr = restart_dict['current_lr']

    def log(self, eval_info, use_wandb=True):

        # if a logging map has not been set
        if self.logging_map is None:
            # create map
            self.logging_map = list(eval_info.keys())
            # write header csv
            with open(self.results_path, 'w') as f:
                # writer.writerow(attrib for i in self.logging_map)
                f.write(','.join(self.logging_map)+'\n')
                f.flush()

        # log info
        if (self.results_path is not None) and (len(self.episode_rewards) > 10):
            # create generator string
            gen_str = ','.join(['{:.2f}' for _ in range(len(self.logging_map))])+"\n"
            # write everything out
            with open(self.results_path, 'a') as f:
                f.write(gen_str.format(*[eval_info[map_key] for map_key in self.logging_map]))
                f.flush()

        # if we are also pushing to wandb
        if use_wandb:
            # go through dict and detach everything
            for key in eval_info.keys():
                if torch.is_tensor(eval_info[key]):
                    eval_info[key] = eval_info[key].detach()
            # log it
            wandb.log(eval_info, step=self.steps)

        # nothing to return
        return None

    def display(self, eval_info):
        print('===============================================================')
        print(self.params.algo_key + ': number of steps {:.0f}, number of time-steps {:.0f}.'.format(eval_info['steps'], eval_info['time_steps']))
        print('mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}'.format( \
            eval_info['r_mean'], eval_info['r_med'], eval_info['r_min'], eval_info['r_max']))
        print('mean/median horizon {:.2f}/{:.2f}, min/max horizon {:.2f}/{:.2f}'.format( \
            eval_info['h_mean'], eval_info['h_med'], eval_info['h_min'], eval_info['h_max']))
        print('current learning rate:', self.lr)
        print('time-stamp:', datetime.datetime.now())
        print('average step time:', mean(self.execution_time))
        print('current beta:', self.algo.actor_critic.beta)
        print('additional info: ', eval_info)
        print('===============================================================')
        return None

    @staticmethod
    def model_init(actor_critic, params):
        # set parameters in trainer class
        if params.algo_key in ['rl-expert']:
            actor_critic.critic_encoder_stopgrad = False
            actor_critic.policy_encoder_stopgrad = True
            actor_critic.sampling_dist = 'expert'
            actor_critic.beta = torch.tensor(0.)
            actor_critic.critic_key = 'agent' # expert is defined as 'agent' here
            actor_critic.use_learned_expert = False
        elif params.algo_key in ['rl-agent']:
            actor_critic.critic_encoder_stopgrad = False
            actor_critic.policy_encoder_stopgrad = True
            actor_critic.sampling_dist = 'agent'
            actor_critic.beta = torch.tensor(0.)
            actor_critic.critic_key = 'agent'
            actor_critic.use_learned_expert = False
        elif params.algo_key == 'ad-agent':
            actor_critic.beta = torch.tensor(params.beta)
            actor_critic.sampling_dist = 'agent'
            actor_critic.critic_key = 'agent'
            actor_critic.use_learned_expert = False
        elif params.algo_key == 'ad-expert':
            actor_critic.beta = torch.tensor(params.beta)
            actor_critic.sampling_dist = 'expert'
            actor_critic.critic_key = 'agent' # expert is defined as 'agent' here
            actor_critic.use_learned_expert = False
        elif params.algo_key == 'a2d-agent':
            actor_critic.beta = torch.tensor(params.beta)
            actor_critic.critic_encoder_stopgrad = False
            actor_critic.policy_encoder_stopgrad = True
            actor_critic.critic_key = 'mixture'
            actor_critic.sampling_dist = 'agent'
            actor_critic.use_learned_expert = True
        elif params.algo_key == 'bc-agent':
            actor_critic.sampling_dist = 'agent'
        elif params.algo_key == 'bc-expert':
            actor_critic.sampling_dist = 'expert'
        else:
            raise Exception('invalid algo_key: '+params.algo_key)
        # return
        return actor_critic

    @staticmethod
    def load_checkpoint(actor_critic, params):

        # load in all available weights
        state_dict = torch.load(params.nn_weights_loc)
        own_state = actor_critic.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            else:
                try:
                    own_state[name].copy_(param)
                    print('loaded:',name)
                except:
                    print('failed to load:', name)
                    continue

        # load in optimizer if policy + full policy
        try:
            actor_critic.load_state_dict(torch.load(params.nn_weights_loc))
        except:
            print('could not load in full optimizer')

    @staticmethod
    def agent_from_params(params, env_gen=None, ac_class=None, add_train_environment=True):
        # settings
        torch.set_default_tensor_type('torch.FloatTensor')
        if ac_class is None:
            ac_class = ActorCritic
        # load in stuff
        p = Trainer(ac_class, params, env_gen=env_gen, add_train_environment=add_train_environment)
        # return the class
        return p
