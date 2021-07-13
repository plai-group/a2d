
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
from carla_a2d.utils.optim_utils import set_optimizer
from carla_a2d.memory.storage import TransitionReplayMemory, StateReplayMemory, DictDataset

class A2D():
    """
    A class used to orginize and train an asymmetric learning agent. It functions
    by alternating between improvement (in this case through an asymmetric expert),
    and a projection step where the agents policy is projected onto the expert.

    Attributes
    ----------
    actor_critic : nn.Module
        Nueral network class which contains both the expert and trainee modules,
        alongside the respective distributions. This class requires act,
        evaluate_actions, get_value, and evaluate_expert_actions.
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
    DAgger_kl(reshaped_obs, detach_encoder)
        Objective used to project agent policy onto the expert. The KL is in reverse,
        meaning that the expectation is with respect to the learned distribution,
        which we found to work better in practice, since no inference takes place on
        top of this solution.
    DAgger_step(sample, detach_encoder, verbose)
        Full DAgger mini-batch training loop used to project agent onto expert
        policy. Returns the average KL across batches for set of updates.
    expert_update_helper(optim, model, rollouts, reshaped_obs, detach_encoder)
        Data formating module that feeds into both the experts policy updates,
        as well as the agent and experts value function updates.
    agent_update_helper(rollouts)
        Helper function to manage updates made to the agents policy (POMDP) via
        DAgger. Also manually shapes rollouts.
    improve(rollouts, policy_update)
        Generic improvement call that manages logging, updates and evaluation for
        the expert policy and both expert and agent value functions.
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
        self.get_optimizer = lambda model: set_optimizer(model, \
            params.lr, params.eps, params.alpha, coeffs, multi_opt=params.single_optim)
        self.optimizer = self.get_optimizer(self.actor_critic)
        # AD stuff
        self.state_buffer = StateReplayMemory(params.AD_buffer_mem)
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
        agent_dist_1 = network1.agent_dist(agent_features1.detach())
        agent_dist_2 = network2.agent_dist(agent_features2.detach())
        # compute kl div
        div = torch.distributions.kl.kl_divergence(agent_dist_1, agent_dist_2).detach()
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
    def ppo_policy_update(self, optim, model, rollouts, reshaped_obs, clip_eps=0.2, actor_epochs=5, detach_encoder=False):

        # obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        # compute AC info -> init from sampled distribution.
        values, action_log_probs, _, _ = model.evaluate_actions(
            reshaped_obs,
            rollouts.actions.view(-1, action_shape),
            rnn_hxs=rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            masks=rollouts.masks[:-1].view(-1, 1), detach_encoder=False, device=self.device)

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
        reshaped_obs.update({'advantages': advantages.view(-1,1).detach(), 'actions': actions.detach(),
                             'action_log_probs':action_log_probs.detach()})
        # update steps
        for _ in range(actor_epochs):
            # create data buffer
            data = DictDataset(reshaped_obs)
            # set data loader
            data_loader = torch.utils.data.DataLoader(data, batch_size=self.params.policy_batch_size, shuffle=True)
            action_loss = 0.
            # iterate
            for r, batch in enumerate(data_loader):
                # get required info
                batch_adv, batch_actions = batch['advantages'], batch['actions']
                old_log_probs = batch['action_log_probs']
                # get log_prob
                _, new_log_probs, dist_entropy, _ = model.evaluate_expert_actions(batch, batch_actions, device=self.device)
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
                # check_grad(model)
                # step the optimizer
                if detach_encoder:
                    optim.step(['expert_policy_params'])
                else:
                    optim.step(['expert_policy_params', 'expert_encoder_params'])
                # get average
                action_loss += loss.detach()
                # # added stopping criteria
                # if self.A2D_kl_constraint(model, model, reshaped_obs) > self.ppo_a2d_constraint:
                #     print('early stopping of ppo taken...')
                #     return action_loss.item() / (r+1)

        # return
        return action_loss.item() / (r+1)

    def ppo_critic_update(self, optim, model, rollouts, reshaped_obs, critic_epochs=3, detach_encoder=False):
        # obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        # compute AC info
        values, action_log_probs, dist_entropy, _ = model.evaluate_actions(
            reshaped_obs, rollouts.actions.view(-1, action_shape),
            device=self.device, detach_encoder=False)
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
            data_loader = torch.utils.data.DataLoader(data, batch_size=self.params.critic_batch_size, shuffle=True)
            value_loss = 0.
            # iterate
            for r, batch in enumerate(data_loader):
                # get target
                batch_taget = batch['value_target'].to(self.device)
                # get values
                new_values = model.get_value(batch, device=self.device)
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

    # imitation learning updates
    def DAgger_kl(self, reshaped_obs, detach_encoder=False):
        # get the distributions
        expert_dist = self.actor_critic.expert_dist(self.actor_critic.base.expert_forward(reshaped_obs)[1].detach())
        agent_dist = self.actor_critic.agent_dist(self.actor_critic.base(reshaped_obs, detach_encoder=detach_encoder)[1])
        # return them
        return torch.distributions.kl.kl_divergence(agent_dist, expert_dist).mean()

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
            # check_grad(self.actor_critic)
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

    # main update helpers
    def expert_update_helper(self, optim, model, rollouts, reshaped_obs, detach_encoder=False):

        # obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        # compute AC info
        values, action_log_probs_, dist_entropy, _ = model.evaluate_actions(
            reshaped_obs,
            rollouts.actions.view(-1, action_shape),
            rnn_hxs=rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            masks=rollouts.masks[:-1].view(-1, 1), device=self.device, detach_encoder=detach_encoder)

        # store policy
        old_policy = deepcopy(self.actor_critic)

        # update policy
        action_loss = self.ppo_policy_update(optim, model, rollouts, reshaped_obs,
            actor_epochs=self.params.policy_updates, clip_eps=self.params.ppo_clip,
            detach_encoder=True)

        # now look at the kl divergence between updates
        kl_div = self.eval_kl(old_policy, model, reshaped_obs)

        # divergence from agent policy
        kl_const = self.A2D_kl_constraint(old_policy, model, reshaped_obs)

        # return
        return action_loss, kl_div.item(), dist_entropy.item(), kl_const.item()

    # main update loop
    def agent_update_helper(self, rollouts):

        # add example to buffer
        self.state_buffer.push(rollouts)

        # init averages
        avg_kl = 0.
        number_of_samples = min(self.state_buffer.get_len(), self.params.AD_batch_size)

        # take optim steps
        for sample_iters in range(self.params.AD_updates_per_batch):
            random_sample = self.state_buffer.sample(number_of_samples,full_buffer=self.params.AD_full_mem)
            avg_kl += self.DAgger_step(random_sample, detach_encoder=True)

        # compute average
        avg_kl /= sample_iters + 1

        # return
        return avg_kl

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
            expert_loss, kl_div, dist_entropy, kl_const = self.expert_update_helper(
                        self.optimizer, self.actor_critic, rollouts, reshaped_obs,
                        detach_encoder=self.actor_critic.policy_encoder_stopgrad)
        else:
            kl_const, kl_div, expert_loss, dist_entropy = 0., 0., 0., 0.

        # update critic (with beta=0 and beta=1)
        start_beta = deepcopy(self.actor_critic.beta)

        # expert critic update
        self.actor_critic.beta = torch.tensor(1.)
        value_loss_1 = self.ppo_critic_update(self.optimizer, self.actor_critic,
                        rollouts, reshaped_obs, critic_epochs=self.params.critic_updates,
                        detach_encoder=self.actor_critic.critic_encoder_stopgrad)

        # agent critic update
        self.actor_critic.beta = torch.tensor(0.)
        value_loss_2 = self.ppo_critic_update(self.optimizer, self.actor_critic,
                        rollouts, reshaped_obs, critic_epochs=self.params.critic_updates,
                        detach_encoder=self.actor_critic.critic_encoder_stopgrad)
        # reset beta
        self.actor_critic.beta = start_beta

        # mode entropy
        _, _, dist_entropy, _ = self.actor_critic.evaluate_actions(
            reshaped_obs, rollouts.actions.view(-1, action_shape),
            rnn_hxs=rollouts.recurrent_hidden_states[0],
            masks=rollouts.masks[:-1], device=self.device)

        # set stuff to return
        return_dict = {'expert_value_loss':value_loss_1, 'expert_loss':expert_loss,
                        'agent_value_loss':value_loss_2,
                        'dist_entropy': dist_entropy}

        # return
        return return_dict

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
