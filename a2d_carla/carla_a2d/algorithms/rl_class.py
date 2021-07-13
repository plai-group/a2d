
#
import time
import numpy as np
import warnings
from copy import deepcopy
import gc
import random
from collections import deque

#
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal

# lib imports
from carla_a2d.utils.optim_utils import set_optimizer
from carla_a2d.memory.storage import StateReplayMemory, DictDataset
from carla_a2d.utils.optim_utils import check_grad

class RL():
    """
    A class used to orginize and train a reinforcement learning agent using
    on-policy optimization, specifically proximal policy optimization.

    Attributes
    ----------
    actor_critic : nn.Module
        Nueral network class which contains both the expert and trainee modules,
        alongside the respective distributions. This class requires act,
        evaluate_actions, get_value.
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
    eval_expert_kl(network2, network1, reshaped_obs)
        This takes two actor_critic networks and computes the kl divergence with
        respect to the expert distribution for a set of states between them.
    eval_agent_kl(network2, network1, reshaped_obs)
        This takes two actor_critic networks and computes the kl divergence with
        respect to the agent distribution for a set of states between them.
    eval_kl(network2, network1, reshaped_obs)
        This takes two actor_critic networks and computes the kl divergence with
        respect to the mixed (between expert and agent following the beta parameter)
        distribution for a set of states between them.
    A2D_kl_constraint(network2, network1, reshaped_obs)
        This takes two actor critic networks and looks at the difference between
        the expert and agent distribution of each respectively. Can be used to
        enforce a specific maximum divergence (similar to TRPO) through the use
        of a linesearch.
    ppo_policy_update(optim, model, rollouts, reshaped_obs, clip_eps, actor_epochs, detach_encoder)
        This update follows the classic proximal policy optimization step, which
        performs mini-batch SGD using a gradient clipped loss, where the clip
        parameter is defined by importance weights. Instead of the TRPO settiing,
        which uses natural policy gradient step with a line-search, PPO takes
        multiple steps and 'throws away' gradient signal in all examples too
        far away from the original behavioral policy. I our case, we replace the
        update, with the experts, thereby replacing the original importance
        weights defined in the paper, by the same clipping mechanism.
    ppo_critic_update(optim, model, rollouts, reshaped_obs, critic_epochs, detach_encoder)
        Mini-batch update of the value function, which seeks to minimize the
        temporal difference error at every stage. SGD is used to reduce memory load.
    improve(rollouts, policy_update)
        Generic improvement call that manages logging, updates and evaluation for
        the expert policy and both expert and agent value functions.
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
        self.get_optimizer = lambda model: set_optimizer(model, \
            params.lr, params.eps, params.alpha, coeffs, multi_opt=params.single_optim)
        self.optimizer = self.get_optimizer(self.actor_critic)
    #
    def eval_expert_kl(self, network2, network1, reshaped_obs):
        # get features
        _, expert_features1, _ = network1.base.expert_forward(reshaped_obs)
        _, expert_features2, _ = network2.base.expert_forward(reshaped_obs)
        # create dist objects
        expert_dist_1 = network1.expert_dist(expert_features1)
        expert_dist_2 = network1.expert_dist(expert_features2)
        # compute kl div
        div = torch.distributions.kl.kl_divergence(expert_dist_1, expert_dist_2)
        # return it
        return div.mean()

    def eval_agent_kl(self, network2, network1, reshaped_obs):
        # get features
        _, agent_features1, _ = network1.base(reshaped_obs)
        _, agent_features2, _ = network2.base(reshaped_obs)
        # create dist objects
        agent_dist_1 = network1.agent_dist(agent_features1)
        agent_dist_2 = network2.agent_dist(agent_features2)
        # compute kl div
        div = torch.distributions.kl.kl_divergence(agent_dist_1, agent_dist_2)
        # return it
        return div.mean()

    def eval_kl(self, network2, network1, reshaped_obs):
        # create data buffer
        data = DictDataset(reshaped_obs)
        # set data loader
        data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
        #
        kl_avg = 0.
        # iterate
        for r, batch in enumerate(data_loader):
            if network2.sampling_dist == 'agent':
                kl_avg += self.eval_agent_kl(network2, network1, batch).detach()
            elif network2.sampling_dist == 'expert':
                kl_avg += self.eval_expert_kl(network2, network1, batch).detach()
            else:
                raise Exception('ahhh')
        # return
        return kl_avg / (r+1)

    def A2D_kl_constraint(self, network2, network1, reshaped_obs):
        # get features
        _, expert_features, _ = network2.base.expert_forward(reshaped_obs)
        _, agent_features, _ = network1.base.forward(reshaped_obs)
        # create dist objects
        expert_dist_ = network2.expert_dist(expert_features)
        agent_dist_ = network1.agent_dist(agent_features)
        # compute kl div
        div = torch.distributions.kl.kl_divergence(expert_dist_, agent_dist_).detach()
        # return it
        return div.mean()

    # ppo updates
    def ppo_policy_update(self, optim, model, rollouts, reshaped_obs, clip_eps=0.2, actor_epochs=5, detach_encoder=True):

        # store policy
        old_policy = deepcopy(self.actor_critic)

        # obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        # compute AC info
        values, action_log_probs, dist_entropy, _ = model.evaluate_actions(
            reshaped_obs,
            rollouts.actions.view(-1, action_shape),
            rnn_hxs=rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            masks=rollouts.masks[:-1].view(-1, 1), detach_encoder=False,
            device=self.device)

        # reshape some things
        values = values.view(num_steps, num_processes, 1)

        # compute value loss
        advantages = rollouts.returns[:-1].detach()
        advantages = advantages - values

        # set actions + log probs
        actions = rollouts.actions.view(-1, action_shape)

        # get normalize loss
        norm_advantages = (advantages - advantages.mean())/(advantages.std()+1e-8)
        # store initial network for check
        starting_model = deepcopy(model)
        # create dict
        reshaped_obs.update({'advantages': norm_advantages.view(-1,1).detach(), 'actions': actions.detach(),
                             'action_log_probs':action_log_probs.detach()})
        # update steps
        for _ in range(actor_epochs):
            # create data buffer
            data = DictDataset(reshaped_obs)
            # set data loader
            data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
            action_loss = 0.
            # iterate
            for r, batch in enumerate(data_loader):
                # get required info
                batch_adv, batch_actions = batch['advantages'], batch['actions']
                old_log_probs = batch['action_log_probs']
                # for k in {'advantages','batch_actions','old_log_probs'}: batch.pop(k, None)
                # get log_prob
                value, new_log_probs, dist_entropy, _ = model.evaluate_actions(batch, batch_actions, device=self.device)
                # get ratios
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1-clip_eps, 1+clip_eps)
                # get surrogate rewards
                unclp_rew = ratio * batch_adv
                clp_rew = clipped_ratio * batch_adv
                # compute surrogate loss
                surrogate = -torch.min(unclp_rew, clp_rew).mean()
                # get loss
                loss = surrogate - self.entropy_coeff * dist_entropy
                # zero out grad loss
                optim.zero_grad()
                # call backward
                loss.backward()
                # add in some gradient normalization
                nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                # step the optimizer
                if detach_encoder:
                    optim.step(['expert_policy_params', 'agent_policy_params'])
                else:
                    optim.step(['expert_policy_params', 'agent_encoder_params',
                                'agent_policy_params', 'expert_encoder_params'])
                # get average
                action_loss += loss.detach()
        # average
        action_loss = action_loss.item() / (r+1)
        # now look at the kl divergence between updates
        kl_div = self.eval_kl(old_policy, model, reshaped_obs)
        # divergence from agent policy
        kl_const = self.A2D_kl_constraint(old_policy, model, reshaped_obs)
        # return
        return action_loss, kl_div.item(), dist_entropy.item(), kl_const.item()

    def ppo_critic_update(self, optim, model, rollouts, reshaped_obs, critic_epochs=3, detach_encoder=False):

        # obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        # compute AC info
        values, action_log_probs, dist_entropy, _ = model.evaluate_actions(
            reshaped_obs,
            rollouts.actions.view(-1, action_shape),
            rnn_hxs=rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            masks=rollouts.masks[:-1].view(-1, 1), device=self.device, detach_encoder=False)

        # reshape some things
        values = values.view(num_steps, num_processes, 1)

        # get original target
        value_target = rollouts.returns[:-1].detach()

        # create dict
        reshaped_obs.update({'value_target': value_target.reshape(-1,1).detach()})

        # update steps
        for _ in range(critic_epochs):
            # create data buffer
            data = DictDataset(reshaped_obs)
            # set data loader
            data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
            value_loss = 0.
            # iterate
            for r, batch in enumerate(data_loader):
                # get target
                batch_taget = batch['value_target']
                # get values
                new_values = model.get_value(batch)
                # get value loss
                value_loss = (batch_taget-new_values).pow(2).mean()
                # first zero out gradient
                optim.zero_grad()
                # value loss
                (value_loss * self.value_loss_coeff).backward()
                # step the optimizer
                if detach_encoder:
                    optim.step(['agent_critic_params', 'expert_critic_params'])
                else:
                    optim.step(['agent_critic_params', 'agent_encoder_params',
                                'expert_critic_params', 'expert_encoder_params'])
            #
            value_loss += value_loss
        # return
        return value_loss.item() / (r+1)

    # main update loop
    def improve(self, rollouts, policy_update=True):

        # obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        # reshaped_obs = rollouts.obs[:-1].view(-1, *obs_shape)
        spaces_names = list(rollouts.obs.keys())
        spaces_shapes = [rollouts.obs[key].size()[2:] for key in spaces_names]
        spaces_dict = {key:value for (key, value) in zip(spaces_names,spaces_shapes)}
        reshaped_obs = {key:rollouts.obs[key][:-1].view(-1, *spaces_dict[key]) for key in spaces_names}

        # should we update the policy
        if policy_update:
            action_loss, kl_div, dist_entropy, kl_const = self.ppo_policy_update(
                        self.optimizer, self.actor_critic, rollouts, reshaped_obs,
                        actor_epochs=self.params.policy_updates, clip_eps=self.params.ppo_clip,
                        detach_encoder=self.actor_critic.policy_encoder_stopgrad)
        else:
            kl_const, kl_div, action_loss, dist_entropy = 0., 0., 0., 0.

        # update critic
        value_loss = self.ppo_critic_update(self.optimizer, self.actor_critic,
                        rollouts, reshaped_obs, critic_epochs=self.params.critic_updates,
                        detach_encoder=self.actor_critic.critic_encoder_stopgrad)

        # set stuff to return
        return_dict = {'value_loss':value_loss, 'action_loss':action_loss, 'dist_entropy':dist_entropy}

        # return
        return return_dict
