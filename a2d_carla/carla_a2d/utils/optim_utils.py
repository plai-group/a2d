
# general stuff
import time
import numpy as np
import warnings
from copy import deepcopy
import gc
from collections import deque
import random

# torch stuff
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class MultiOptimizer(object):

    """
    This manages a dictionary of optimizers and steps them based upon set of
    key words. This allows for the correct update of optimization algorithms
    which rely on statistical information (such as adam) in the presence of
    learning approaches which iteratively detach parts of the computation
    graph. This is surprisingly important in actor critic algorithms which
    share encoders, where only one loss will apply updates to the encoder on
    any given pass.

    Attributes
    ----------
    op: dict
    Dictionary of optimizers which will be used in each update.

    Methods
    -------
    zero_grad(key_subset)
    applies zero grad to all or some of the optimizers from the dictionary.

    step(key_subset)
    applies step to all or some of the optimizers from the dictionary.

    load_state_dict(state_dict_dict)
    Updates the state-dictionary of optimizer.

    state_dict()
    Returns the state-dictionary of optimizer.

    get_lr()
    Returns the learning rates of individual optimizers.

    set_lr(new_lr, lr_keys)
    Updates the learning rates of individual optimizers.

    scale_lr(scale, key_subset)
    Scales the learning rates of individual optimizers.
    """

    def __init__(self, op):
        self.optimizers_dict = op
        if 'full' in op.keys():
            self.single_opt = True
        else:
            self.single_opt = False

    def zero_grad(self, key_subset=None):
        if self.single_opt:
            self.optimizers_dict['full'].zero_grad()
        elif key_subset is None:
            for key in self.optimizers_dict.keys():
                self.optimizers_dict[key].zero_grad()
        else:
            for key in key_subset:
                if key in self.optimizers_dict.keys():
                    self.optimizers_dict[key].zero_grad()

    def step(self, key_subset=None):
        if self.single_opt:
            self.optimizers_dict['full'].step()
            return
        elif key_subset is None:
            for key in self.optimizers_dict.keys():
                self.optimizers_dict[key].step()
        else:
            for key in key_subset:
                if key in self.optimizers_dict.keys():
                    self.optimizers_dict[key].step()

    def load_state_dict(self, state_dict_dict):
        if self.single_opt:
            self.optimizers_dict['full'].load_state_dict(state_dict_dict['full'])
            return
        else:
            for key in state_dict_dict:
                self.optimizers_dict[key].load_state_dict(state_dict_dict[key])

    def state_dict(self):
        if self.single_opt:
            return {'full': self.optimizers_dict['full'].state_dict()}
        state_dict_dict = {}
        for key in self.optimizers_dict.keys():
            if key in self.optimizers_dict.keys():
                state_dict_dict[key] = self.optimizers_dict[key].state_dict()
        return state_dict_dict

    def get_lr(self):
        self.lr = {key:[group['lr'] for group in self.optimizers_dict[key].param_groups] for key in self.optimizers_dict.keys()}
        return self.lr

    def set_lr(self, new_lr, lr_keys=None):
        assert new_lr is not None
        if self.single_opt:
            for group in self.optimizers_dict['full'].param_groups:
                group['lr'] = new_lr
            self.lr = {key:[group['lr'] for group in self.optimizers_dict[key].param_groups] for key in self.optimizers_dict.keys()}
            return

        if lr_keys is None:
            for key in self.optimizers_dict.keys():
                for group in self.optimizers_dict[key].param_groups:
                    group['lr'] = new_lr
        else:
            lr_keys = {key:lr_keys[key] for key in lr_keys.keys() if key in self.optimizers_dict}
            for key in self.optimizers_dict.keys():
                for group, new_lr in zip(self.optimizers_dict[key].param_groups, lr_keys[key]):
                    group['lr'] = new_lr
        self.lr = {key:[group['lr'] for group in self.optimizers_dict[key].param_groups] for key in self.optimizers_dict.keys()}

    def scale_lr(self, scale, key_subset=None):
        if self.single_opt:
            for group in self.optimizers_dict['full'].param_groups:
                group['lr'] *= scale
        if key_subset is None:
            for key in self.optimizers_dict.keys():
                for group in self.optimizers_dict[key].param_groups:
                    group['lr'] *= scale
        else:
            raise NotImplementedError
        self.lr = {key:[group['lr'] for group in self.optimizers_dict[key].param_groups] for key in self.optimizers_dict.keys()}

def generate_enumerator(dict):
    """ Wraps list in generator object following input for pytorch optim. """
    return [dict[key] for key in dict.keys()].__iter__()

def set_optimizer(actor_critic, lr, eps, alpha, coeffs, multi_opt=True):
    """ Creates dictionary of optimizers feeding into MultiOptimizer class. """
    # all names
    all_params = {name:params for (name, params) in actor_critic.named_parameters()}
    # split
    agent_params = {name:params for (name, params) in actor_critic.named_parameters() if ('agent' in name)}
    expert_params = {name:params for (name, params) in actor_critic.named_parameters() if ('expert' in name)}
    # critics
    expert_critic_params = {name:expert_params[name] for name in expert_params.keys() if ('critic' in name or 'q_function' in name)}
    agent_critic_params = {name:agent_params[name] for name in agent_params.keys() if ('critic' in name or 'q_function' in name)}
    # encoders
    expert_encoder_params = {name:expert_params[name] for name in expert_params.keys() if ('encoder' in name)}
    agent_encoder_params = {name:agent_params[name] for name in agent_params.keys() if ('encoder' in name)}
    # policies
    expert_policy_params = {name:expert_params[name] for name in expert_params.keys() \
            if ((name not in expert_critic_params) and (name not in expert_encoder_params))}
    agent_policy_params = {name:agent_params[name] for name in agent_params.keys() \
            if ((name not in agent_critic_params) and (name not in agent_encoder_params))}
    # print('ahhhh', agent_dynamics_params, expert_encoder_params)
    # now set optimizers
    param_set = {'expert_critic_params': optim.Adam(
                    generate_enumerator(expert_critic_params), coeffs['critic']*lr, eps=eps),
                 'agent_critic_params': optim.RMSprop(
                    generate_enumerator(agent_critic_params), coeffs['critic']*lr, eps=eps, alpha=alpha, weight_decay=1e-5),
                 'agent_encoder_params': optim.RMSprop(
                    generate_enumerator(agent_encoder_params), coeffs['encoder']*lr, eps=eps, alpha=alpha, weight_decay=1e-5),
                 'expert_encoder_params': optim.RMSprop(
                    generate_enumerator(expert_encoder_params), coeffs['encoder']*lr, eps=eps, alpha=alpha, weight_decay=1e-5),
                 'expert_policy_params': optim.Adam(
                    generate_enumerator(expert_policy_params), coeffs['policy']*lr, eps=eps),
                 'agent_policy_params': optim.RMSprop(
                    generate_enumerator(agent_policy_params), coeffs['policy']*lr, eps=eps, alpha=alpha, weight_decay=1e-5),
                    }
    # wrap optimizer(s)
    if multi_opt:
        optimizer = MultiOptimizer(param_set)
    else:
        optimizer = MultiOptimizer({'full': optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha, weight_decay=1e-5)})
    # check that all parameters are attached
    update_params = {**expert_critic_params, **agent_critic_params, **expert_encoder_params, **agent_encoder_params, \
            **expert_policy_params, **agent_policy_params}
    assert set(update_params.keys()) == set(all_params.keys())
    # Ahhhhh
    return optimizer

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr, multiple_opt=False):
    """Decreases the learning rate linearly"""
    new_lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    old_lr = initial_lr - (initial_lr * ((epoch - 1) / float(total_num_epochs)))
    if not multiple_opt:
        for param_group in optimizer.param_groups:
            if new_lr <= 0.:
                pass
                return old_lr, optimizer
            else:
                param_group['lr'] = (param_group['lr'] / old_lr) * new_lr
                return new_lr, optimizer
    else:
        for key in optimizer.optimizers_dict.keys():
            for group in optimizer.optimizers_dict[key].param_groups:
                if new_lr <= 0.:
                    pass
                else:
                    group['lr'] = (group['lr'] / old_lr) * new_lr
        return new_lr, optimizer

def pretrain_loader(self, params):
    """ Loads old model and optimizer """
    # load in all available weights
    state_dict = torch.load(params.nn_weights_loc)
    own_state = self.algo.actor_critic.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
             continue
        else:
            try:
                own_state[name].copy_(param)
                print('loaded:',name)
            except:
                print('failed to load:',name)
                continue
    # load in optimizer if policy + full policy
    try:
        self.algo.actor_critic.load_state_dict(torch.load(params.nn_weights_loc))
    except:
        print('could not load in full optimizer')

def check_grad(model, verbose=False, non_zero=True):

    """ debugging tool. """

    print('looking at named gradients.....')
    for name, param in model.named_parameters():
        if param.grad is not None:
            if non_zero and torch.sum(param.grad) > 0.:
                print(name, torch.sum(param.grad))
            elif not non_zero:
                print(name, torch.sum(param.grad))
        else:
            pass

    if verbose:
        print('looking at all parameters.....')
        for k, v in model.state_dict().items():
            print(k, type(v))

        if set([k[0] for k in model.state_dict().items()]) == set([k[0] for k in model.named_parameters()]):
            print('all parameters accounted for.')
        else:
            print('some parameters not accounted for.')

    return None
