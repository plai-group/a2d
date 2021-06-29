# This code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Licence.  The full licence and information is available at:
# https://github.com/plai-group/a2d/blob/master/LICENCE.md
# © Andrew Warrington, J. Wilder Lavington, Adam Scibior, Mark Schmidt & Frank Wood.
# Accompanies the paper: Robust Asymmetric Learning in POMDPs, ICML 2021.

import argparse
import torch
import os
import git
import socket
import timeit
from datetime import datetime
from platform import system

if ('Linux' in system()) or ('linux' in system()):
    cluster = True
else:
    cluster = False
machine = socket.gethostname()


def get_args():
    """
    Get the standard arguments for running Q experiments from the supplementary materials.

    This is configured for reproduction of experimental results from the paper as quickly and simply
    as possible.  Set the flag for which methods you want to run.  Each method can be run max once for
    the selected hyperparameter settings.
    :return:
    """

    # Define the arg parser.
    parser = argparse.ArgumentParser(description='A2D arguments.')

    # Which methods to run.
    parser.add_argument('--rl-state',               default=int(False), metavar='N', help='01', type=int)
    parser.add_argument('--rl-partial-state',       default=int(False), metavar='N', help='04', type=int)
    parser.add_argument('--a2d-partial-state',      default=int(False), metavar='N', help='06', type=int)

    # # These are the three OSC environments.
    # parser.add_argument('--env-name', type=str, default="MiniGrid-TigerDoorEnvOSC-v1", metavar='G', help='')
    # parser.add_argument('--env-name', type=str, default="MiniGrid-TigerDoorEnvOSC-v2", metavar='G', help='')
    parser.add_argument('--env-name', type=str, default="MiniGrid-TigerDoorEnvOSC-v3", metavar='G', help='')

    # Main A2D/RL/AIL settings.
    parser.add_argument('--iters', type=int, default=201, metavar='N', help='iterations to run.')
    parser.add_argument('--batch-size', type=int, default=2000, metavar='N', help='batch size per iteration.')
    parser.add_argument('--buffer-size', type=int, default=5000, metavar='G', help='buffer size for A2D/AIL')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N', help='interval between training status logs (default: 10)')
    parser.add_argument('--lambda_a2d', type=float, default=0.5, metavar='G', help='GAE parameter lambda to use in A2D.')
    parser.add_argument('--lambda_rl', type=float, default=0.5, metavar='G', help='GAE parameter lambda to use in straight RL.')
    parser.add_argument('--max-kl', type=float, default=0.001, metavar='G', help='KL step in TRPO.')
    parser.add_argument('--USE-Q', type=int, default=int(False), metavar='G', help='Use a Q fn.  Only works for partial observe.')
    parser.add_argument('--value-calc', type=str, default="time", metavar='G', help='loss fn for learning value fn, {"time", "gae"}.')
    parser.add_argument('--regularize-adv', type=int, default=int(False), metavar='G', help='directly regularize advantages, or, regularize surrogate loss?')
    parser.add_argument('--rl-entropy-reg', type=float, default=0.02, metavar='G', help='entropy regularization to apply.')
    parser.add_argument('--a2d-entropy-reg', type=float, default=0.02, metavar='G', help='entropy regularization to apply.')
    parser.add_argument('--beta-decay', type=float, default=0.0, metavar='G', help='Rate at which to decay mixture parameter beta.')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N', help='interval between training status logs (default: 10)')
    parser.add_argument('--initialization', type=str, default="fixed", metavar='G', help='How to initialize policy networks.')
    parser.add_argument('--eval-batch-size', type=int, default=2000, metavar='G', help='interactions to evaluate over.')

    # All of the other hyperparameters under the sun...
    parser.add_argument('--max-proj-iter', type=int, default=2, metavar='G', help='number of projection steps to use each iteration.')
    parser.add_argument('--val-epochs', type=int, default=25, metavar='G', help='Epochs to use at each iteration learning val fn.')
    parser.add_argument('--q-epochs', type=int, default=25, metavar='G', help='Epochs to use at each iteration learning q fn.')
    parser.add_argument('--mode', type=str, default='A2D', metavar='G', help='default mode to run in.  normally overwritten anyway.')
    parser.add_argument('--cotrain', type=int, default=1, help='Co-train MDP and POMDP')
    parser.add_argument('--data-aug', type=int, default=1, metavar='N', help='include data augmentation?')
    parser.add_argument('--pretraining-projection-steps', type=int, default=1, metavar='N', help='How many pretraining steps (default: 1)')
    parser.add_argument('--dagger-mini-batch-size', type=int, default=64, metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G', help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--rl-lr', type=float, default=7e-4, help='learning rate for rl agents (default: 7e-4)')
    parser.add_argument('--val-lr', type=float, default=7e-4, metavar='G', help='lr for value functions (default: 7e-4)')
    parser.add_argument('--q-lr', type=float, default=3e-4, metavar='G', help='lr for q functions (default: 3e-4)')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate for imitation learning (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.995, metavar='G', help='discount factor (default: 0.995)')
    parser.add_argument('--frame-stack', type=int, default=1, metavar='G', help='Consecutive frames to stack.  Need to double check ReplayBuffer for > 1.')
    parser.add_argument('--resize', type=int, default=42, metavar='G', help='Default size of image observations.')
    parser.add_argument('--policy-net-type', type=str, default="DiscPolicy", metavar='G', help='Policy network type.')
    parser.add_argument('--value-net-type', type=str, default="ValueNet", help='Value fn type.')
    parser.add_argument('--render-type', type=str, default='partial_state', metavar='G', help='Default render type. ')
    parser.add_argument('--env-tag', type=str, default='minigrid', metavar='G', help='Tag for environment.  (default "minigrid")')
    parser.add_argument('--tensor-default', type=str, default='torch.DoubleTensor', help='tensor precision')
    parser.add_argument('--folder-extension', type=str, default=datetime.now().strftime('%Y_%m_%d__%H_%M_%S'), metavar='N', help='unique extension for results folder.')
    parser.add_argument('--vf-steps', type=int, default=1, help='Not used.')
    parser.add_argument('--tau', type=float, default=0.97, metavar='G', help='gae tau parameter (default: 0.97)')
    parser.add_argument('--damping', type=float, default=1e-1, metavar='G', help='damping (default: 1e-1)')
    parser.add_argument('--seed', type=int, default=543, metavar='N', help='Seed.  Actually only used for separating runs now.')
    parser.add_argument('--log-dir', type=str, default='./tests/results', metavar='P', help='base dir for results logging.')
    parser.add_argument('--DO-TRAJECTORY-RENDER', type=int, default=int(True), metavar='G', help='Save pictures? Slows down run.')
    parser.add_argument('--very-verbose', type=int, default=int(False), metavar='N', help='generate lots of output.')

    # Menial TRPO settings.
    parser.add_argument('--rl-method', type=str, default='trpo', metavar='G', help='MDP refinement method.')
    parser.add_argument('--anneal-lr', type=int, default=0, metavar='G', help='Anneal LR in learning.  Not currently used.')
    parser.add_argument('--num-minibatches', type=int, default=32, metavar='G', help='Number of minibatches to construct in valfn epoch.')
    parser.add_argument('--clip-eps', type=float, default=0.2, metavar='G', help='Clipping parameter in val fn learning.')
    parser.add_argument('--value-clipping', type=int, default=1, metavar='G', help='Clip in val fn learning?')
    parser.add_argument('--fisher-frac-samples', type=float, default=0.1, metavar='G', help='TRPO fraction samples to use in f est.')
    parser.add_argument('--cg-steps', type=int, default=10, metavar='G', help='Number of steps in the conj gradient.')
    parser.add_argument('--max-backtrack', type=int, default=10, metavar='G', help='Number of steps in backtrack line search.')
    parser.add_argument('--kl-approximation-iters', type=int, default=-1, metavar='G', help='Not used.')
    parser.add_argument('--eps', type=float, default=1e-5, help='Epsilon for optimisers (default: 1e-5)')

    # These aren't used in Q.
    parser.add_argument('--preenc-train-new-encoder', type=int, default=int(True), metavar='G', help='')
    parser.add_argument('--preenc-encoder-steps', type=int, default=100, metavar='G', help='')
    parser.add_argument('--preenc-encoder-samples', type=int, default=20000, metavar='G', help='')
    parser.add_argument('--preenc-test-encoder', type=int, default=int(True), metavar='G', help='')

    # parse your arguments
    args = parser.parse_args()

    # Inscribe default entropy reg and lambda values.
    args.lambda_ = args.lambda_rl
    args.entropy_reg = args.rl_entropy_reg

    # Write in other menial stuff.
    args.cotrain = bool(args.cotrain)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.cluster = cluster
    args.machine = machine
    args.pretrain_batch_size = args.buffer_size
    args._st = timeit.default_timer()
    args.OSC = True
    args.save_as_expert = False
    args.t = args.batch_size            # Batch size in original RL code...
    args.train_steps = args.iters       # Number of iters in original RL code...
    args.max_kl_final = args.max_kl     # Removed LR annealing.

    # Grab some git information.
    try:
        args.git_commit = git.Repo(search_parent_directories=True).head.object.hexsha
        args.git_branch = git.Repo(search_parent_directories=True).active_branch
        args.git_is_dirty = git.Repo(search_parent_directories=True).is_dirty()
    except:
        print('Failed to grab git info...')
        args.git_commit = 'NoneFound'
        args.git_branch = 'NoneFound'
        args.git_is_dirty = 'NoneFound'

    return args

