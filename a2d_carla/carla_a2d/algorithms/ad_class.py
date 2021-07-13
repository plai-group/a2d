
# general imports
import time
import numpy as np
import warnings
from copy import deepcopy
import gc
from collections import deque
import random

# torch requires
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal

# lib imports
from carla_a2d.utils.optim_utils import set_optimizer, check_grad
from carla_a2d.memory.storage import TransitionReplayMemory, StateReplayMemory, DictDataset

class AD():
    """
    A class used to orginize and train an a generic actor critic class as
    defined in models with respect to a fixed set of expert actions. This
    Algorithm uses mini-batch SGD, a DAgger update, and a reverse KL loss.

    Attributes
    ----------
    actor_critic : nn.Module
        Nueral network class which contains both the expert and trainee modules,
        alongside the respective distributions. This class requires act,
        evaluate_actions.
    params : Namespace
        ArgParse Object which contains the set of parameters specifciied on
        initialization. These include general RL parameters, as well as optimization
        parameters, and logging info.
    device : str
        The device which should house the mini-batch as well as the associated
        computation graph used for the pytorch backwards pass. By default it is
        set to be the second GPU, however this is masked by agent.

    Methods
    -------
    DAgger_kl(reshaped_obs, detach_encoder)
        Objective used to project agent policy onto the expert. The KL is in reverse,
        meaning that the expectation is with respect to the learned distribution,
        which we found to work better in practice, since no inference takes place on
        top of this solution.
    DAgger_step(sample, detach_encoder, verbose)
        Full DAgger mini-batch training loop used to project agent onto expert
        policy. Returns the average KL across batches for set of updates.
    agent_update_helper(rollouts)
        Helper function to manage updates made to the agents policy (POMDP) via
        DAgger. Also manually shapes rollouts.
    project(rollouts)
        Generic project call that manages logging, updates and evaluation for
        the agent policy through the use of DAgger updates wrt the learned expert.
    """
    def __init__(self, actor_critic, params, device='cuda:1'):
        # main inits
        self.actor_critic = actor_critic
        self.params = params
        # coefficients
        self.value_loss_coeff = params.value_loss_coeff
        self.action_loss_coeff = params.action_loss_coeff
        self.entropy_coeff = params.entropy_coeff
        self.encoder_coeff = params.encoder_coeff
        # set the maximum
        self.max_grad_norm = params.max_grad_norm
        self.device = device
        # set coeffs for individual scaling
        coeffs = {'critic': params.value_loss_coeff,
                  'encoder': params.encoder_coeff,
                  'policy': params.action_loss_coeff}
        # optimizer
        self.lr = params.lr
        self.optimizer = set_optimizer(self.actor_critic, params.lr, params.eps, params.alpha, coeffs, multi_opt=params.single_optim)
        # AD stuff
        self.state_buffer = StateReplayMemory(params.AD_buffer_mem)

    # imitation learning updates
    def DAgger_kl(self, reshaped_obs, detach_encoder=False):
        # get the distributions
        fixed_expert_dist = self.actor_critic.fixed_expert(reshaped_obs['fixed_expert_actions'], self.device)
        # grab evals
        if self.params.algo_key == 'ad-agent':
            agent_dist = self.actor_critic.agent_dist(self.actor_critic.base(reshaped_obs, detach_encoder=detach_encoder)[1])
        elif self.params.algo_key == 'ad-expert':
            agent_dist = self.actor_critic.expert_dist(self.actor_critic.base.expert_forward(reshaped_obs, detach_encoder=detach_encoder)[1])
        else:
            raise Exception('provide valid key',self.params.algo_key,'not valid.')
        # return them
        return torch.distributions.kl.kl_divergence(agent_dist, fixed_expert_dist).mean()

    def DAgger_step(self, sample, detach_encoder=False, verbose=False):
        # set detachh setup
        detach_setup = ['agent_policy_params', 'expert_policy_params'] if detach_encoder else None
        # create data set
        data = DictDataset(sample) #torch.utils.data.TensorDataset(sample)
        # set data loader
        data_loader = torch.utils.data.DataLoader(data, batch_size=self.params.AD_batch_size, shuffle=True)
        # init
        avg_kl = 0.
        # iterate
        for r, batch in enumerate(data_loader):
            avg_kl_ = self.DAgger_kl(batch, detach_encoder=detach_encoder)
            # first zero out gradient
            self.optimizer.zero_grad()
            # which to call backwards on.
            avg_kl_.backward()
            # step the optimizer
            self.optimizer.step(detach_setup)
            # get averaging info
            avg_kl += avg_kl_.detach()
        # return everything
        return avg_kl.detach().item() / (r+1)

    # main update loop
    def agent_update_helper(self, rollouts):

        # add example to buffer
        self.state_buffer.push(rollouts)

        # init averages
        avg_kl = 0.
        number_of_samples = min(self.state_buffer.get_len(), self.params.AD_batch_size)

        # take optim steps
        for sample_iters in range(self.params.AD_updates_per_batch):
            random_sample = self.state_buffer.sample(number_of_samples, full_buffer=self.params.AD_full_mem)
            avg_kl += self.DAgger_step(random_sample, detach_encoder=False)

        # compute average
        avg_kl /= sample_iters + 1

        # return
        return avg_kl

    # main update loop
    def project(self, rollouts):

        # obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        # reshaped_obs = rollouts.obs[:-1].view(-1, *obs_shape)
        spaces_names = list(rollouts.obs.keys())
        spaces_shapes = [rollouts.obs[key].size()[2:] for key in spaces_names]
        spaces_dict = {key:value for (key, value) in zip(spaces_names,spaces_shapes)}
        reshaped_obs = {key:rollouts.obs[key][:-1].view(-1, *spaces_dict[key]) for key in spaces_names}

        #
        agent_loss = self.agent_update_helper(rollouts)

        # mode entropy
        _, _, dist_entropy, _ = self.actor_critic.evaluate_actions(
            reshaped_obs, rollouts.actions.view(-1, action_shape),
            rnn_hxs=rollouts.recurrent_hidden_states[0],
            masks=rollouts.masks[:-1], device=self.device)

        # set stuff to return
        return_dict = {'agent_loss': agent_loss, 'dist_entropy': dist_entropy}

        # return
        return return_dict
