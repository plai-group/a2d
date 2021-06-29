# This code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Licence.  The full licence and information is available at:
# https://github.com/plai-group/a2d/blob/master/LICENCE.md
# © Andrew Warrington, J. Wilder Lavington, Adam Scibior, Mark Schmidt & Frank Wood.
# Accompanies the paper: Robust Asymmetric Learning in POMDPs, ICML 2021.

# general imports
import os
from a2d.util.replay_buffer import ReplayBuffer
from a2d.util.helpers import do_log_header


def init_replay_buffer(args):
    """
    AW - initialize the replay buffer.
    :param args:
    :return:
    """
    return ReplayBuffer(args.dagger_mini_batch_size, args.buffer_size)


def init_logger(args_):
    """
    AW - initialize the logger which will output results.
    :param args_:
    :return:
    """
    # Make directory.
    try: os.makedirs(args_.log_dir)
    except: pass

    # Open a file handle which we can just write to on demand.
    results = open(os.path.join(args_.log_dir, 'results.csv'), 'w')

    # Write the results header.
    do_log_header(results)

    # Return the open file handle.
    return results


def init_mdp_net(args, env, mdp_policy_class, value_class):
    """
    AW - Initialize the expert network.

    NOTE - this code is depreciated and the networks are just initialized by
    directly calling the network class.

    :param args:
    :param env:
    :param mdp_policy_class:
    :param value_class:
    :return:
    """
    num_inputs = env.state.shape[0]
    num_actions = env.action_space.shape[0]
    mdp_policy = mdp_policy_class(num_inputs, num_actions)
    value_net = value_class(num_inputs)
    return mdp_policy, value_net


# def init_pomdp_net(env, img_stack, img_size, pomdp_policy_class):
#     """
#     AW - initialize the learner.  This assumes that the learner takes an image as
#     input, otherwise, just initialize the learner directly.
#
#     NOTE - This code is depreciated now.
#     :param env:
#     :param img_stack:
#     :param img_size:
#     :param pomdp_policy_class:
#     :return:
#     """
#     num_actions = env.action_space.shape[0]
#
#     # Initialize the network.  Assumes 3-channel inputs.
#     pomdp_policy = pomdp_policy_class((3*img_stack, img_size, img_size), num_actions)
#
#     return pomdp_policy
