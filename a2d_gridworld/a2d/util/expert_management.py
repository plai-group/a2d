# This code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Licence.  The full licence and information is available at:
# https://github.com/plai-group/a2d/blob/master/LICENCE.md
# © Andrew Warrington, J. Wilder Lavington, Adam Scibior, Mark Schmidt & Frank Wood.
# Accompanies the paper: Robust Asymmetric Learning in POMDPs, ICML 2021.

# general imports
import torch
import warnings
from copy import deepcopy
import numpy as np

from a2d.util.inits import init_mdp_net


def load_expert(core_dir, policy_net, value_net):
    print('Loading expert from: {}'.format(core_dir))
    expert_dict = torch.load(core_dir)
    try:
        expert_performance = expert_dict['expert_performance']
    except:
        expert_performance = None
    # policy
    expert_policy_dict = expert_dict['expert_network']
    expert_policy = deepcopy(policy_net)
    expert_policy.load_state_dict(expert_policy_dict)
    expert_policy.expert_performance = expert_performance
    expert_policy.render_type = expert_dict['expert_render_type']
    expert_policy.env_tag = expert_dict['expert_env_tag']
    expert_policy.env_name = expert_dict['expert_env_name']
    # vf
    vf_dict = expert_dict['expert_value_net']
    expert_vf = deepcopy(value_net)
    expert_vf.load_state_dict(vf_dict)

    try:
        print('Expert performance: {:8.2f}.'.format(expert_performance))
    except:
        print('Expert performance: N/A')

    return expert_policy, expert_vf


# grab / train an expert prior
def get_expert_network(args, env, create_new_expert=False, dir_save_to='test_1/',file_name='test_1.pt'):

    """ INITS """
    # suppress warngins
    warnings.filterwarnings('ignore')

    # AW - slightly hacky way of digging out reasonable networks.
    # get class info
    from a2d.models.models import policy_net_with_name, value_net_with_name
    Policy = policy_net_with_name(args.policy_net_type)
    Value = value_net_with_name(args.value_net_type)

    policy_net, value_net = init_mdp_net(args, env, Policy, Value)

    if args.cotrain:
        # If we are going to co-train, just get the raw networks.
        print('COTRAINING networks, skipping loading expert.')
        policy_net.expert_performance = -90210
        policy_net.render_type = args.render_type
        policy_net.env_name = args.env_name
        policy_net.env_tag = args.env_tag
        return policy_net, value_net

    core_dir = dir_save_to + file_name

    try:
        expert_policy, expert_vf = load_expert(core_dir, policy_net, value_net)
    except Exception as err:
        print('Loading error: ' + str(err))
        print('No expert available at: '+core_dir)
        raise RuntimeError

    return expert_policy, expert_vf
