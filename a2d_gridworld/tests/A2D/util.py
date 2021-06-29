# This code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Licence.  The full licence and information is available at:
# https://github.com/plai-group/a2d/blob/master/LICENCE.md
# © Andrew Warrington, J. Wilder Lavington, Adam Scibior, Mark Schmidt & Frank Wood.
# Accompanies the paper: Robust Asymmetric Learning in POMDPs, ICML 2021.

import os
import pickle
import gym
import shutil

from copy import deepcopy
from datetime import datetime

import a2d.A2Dagger as A2Dagger
from a2d.util.helpers import *
from a2d.util.expert_management import get_expert_network, load_expert
from a2d.environments.env_wrap import EnvWrapper
from a2d.models.models import DiscPolicy, ValueDenseNet
import a2d.environments.minigrid.gym_minigrid


def make_logging(default_args):
    """
    Set up the initial logging directory and inscribe in to the args.
    Also set up a redirect (tee) to capture output and log to file.
    :param default_args:
    :return:
    """
    # Dump args.
    default_args.base_dir = 'tests/results/AdaptAsymDagger/' + default_args.env_name + "/"
    try:
        os.mkdir(default_args.base_dir)
    except:
        pass

    default_args.base_dir += default_args.folder_extension
    try:
        os.mkdir(default_args.base_dir)
    except:
        pass

    # Dump some args.
    try:
        with open(default_args.base_dir + "/args.p", 'wb') as f:
            pickle.dump(default_args, f)
    except:
        pass

    # Make redirect.
    tee = Tee('tests/results/AdaptAsymDagger/' + default_args.env_name + "/" +
              default_args.folder_extension + "/report_" + str(default_args.seed) + ".txt")
    print(default_args)
    print('Start time: ' + datetime.now().strftime('%Y_%m_%d__%H_%M_%S'))

    return default_args, tee


def set_single_dir(_mode, _args):
    """
    Make the single logging directory inside the experimental file.
    :param _mode:
    :param _args:
    :return:
    """
    _args.log_dir = _args.base_dir + "/" + _mode + "/" + str(_args.seed)
    try:
        os.makedirs(_args.log_dir)
    except:
        pass
    with open(_args.log_dir + "/args.p", 'wb') as f:
        pickle.dump(_args, f)
    try:
        os.mkdir(_args.log_dir + '/im0')
    except:
        pass
    return _args


def do_rl(args, _render_type, _frame_stack, _return_type, _run_rl=True):
    """
    Configure the experiment for straightforward RL.
    :param args:
    :param _render_type:
    :param _frame_stack:
    :param _return_type:
    :param _run_rl:
    :return:
    """
    rl_args = deepcopy(args)
    rl_args.mode = 'RL'
    rl_args.cotrain = False
    rl_args.render_type = _render_type
    rl_args.frame_stack = _frame_stack
    rl_args.entropy_reg = rl_args.rl_entropy_reg

    env = EnvWrapper(gym.make(rl_args.env_name), rl_args)
    observation_space = env.my_render()

    if rl_args.render_type == 'state':
        expert_pol = DiscPolicy(observation_space.shape, env.action_space.shape[0])
        expert_val = ValueDenseNet(observation_space.shape)
        learner_pol = None
        learner_val = None

    else:
        expert_pol = None
        expert_val = None
        learner_pol = DiscPolicy(observation_space.shape, env.action_space.shape[0])
        learner_val = ValueDenseNet(observation_space.shape)

    rl_args = set_single_dir(rl_args.mode + '_' + rl_args.render_type, rl_args)

    if _run_rl:
        trainer = A2Dagger.AdpAsymDagger(rl_args, img_stack=rl_args.frame_stack, img_size=rl_args.resize)
        trainer.train(expert_pol, expert_val, learner_pol, learner_val, pre_train=False)

    return rl_args, env


def do_a2d(args, _mode, _render_type, _frame_stack, _cotrain):
    """

    :param args:
    :param _mode:
    :param _render_type:
    :param _frame_stack:
    :param _cotrain:
    :return:
    """
    # Lets re-make the arguments disctionary.
    a2d_args = deepcopy(args)
    a2d_args.mode = _mode
    a2d_args.cotrain = _cotrain
    a2d_args.render_type = _render_type
    a2d_args.frame_stack = _frame_stack

    # We will often use a different entropy regularisation in A2D.
    if _mode == 'A2D':
        a2d_args.entropy_reg = a2d_args.a2d_entropy_reg
        a2d_args.lambda_ = a2d_args.lambda_a2d

    # Make an environment so we can fetch sizes of things.
    env = EnvWrapper(gym.make(a2d_args.env_name), a2d_args)
    observation_space = env.my_render()

    # Fetch the learner policies we will use.
    learner_pol = DiscPolicy(observation_space.shape, env.action_space.shape[0])
    learner_val = ValueDenseNet(observation_space.shape)

    # Fetch the appropriate expert policies.
    # If we are doing A2D, cotrain will disable loading the agent and will
    # simple define the networks instead.
    expert_pol, expert_val = get_expert_network(a2d_args, env, dir_save_to=a2d_args.expert_location,
                                                file_name='/expert.pt', create_new_expert=False)

    # Make the output directory for this experiment.
    a2d_args = set_single_dir(a2d_args.mode + '_' + a2d_args.render_type, a2d_args)

    # Make the A2D class.
    trainer = A2Dagger.AdpAsymDagger(a2d_args, img_stack=a2d_args.frame_stack, img_size=a2d_args.resize)

    # Provide the networks to the trainer and lets fly!
    trainer.train(expert_pol, expert_val, learner_pol, learner_val)


def do_advanced_rl(args, _render_type, _frame_stack, _return_type, _ete):
    """
    Set up for ARL and PreEnc.  These are a little more involved to set up.
    :param args:
    :param _render_type:
    :param _frame_stack:
    :param _return_type:
    :param _ete:
    :return:
    """
    # Set up using asymmetric RL, or, using a pretrained encoder.
    arl_args = deepcopy(args)
    arl_args.render_type = _render_type
    arl_args.return_type = _return_type
    arl_args.frame_stack = _frame_stack
    arl_args.cotrain = 0
    arl_args.entropy_reg = arl_args.rl_entropy_reg

    # Are we using a pretrained encoder, or asymmetric RL?
    if _ete:
        # Pretrained encoder.
        arl_args.mode = 'ete'

        if 'Tiger' in arl_args.env_name:
            # NOTE - we had to drop the KL for TigerDoor.
            arl_args.max_kl = 0.005
            arl_args.max_kl_final = 0.005
            print('NOTE - hard setting KL for TD+PreEnc.  0.01 led to unstable convergence.')

    else:
        # Asymmetric RL.
        arl_args.mode = 'ARL'

    # Set the directory we will save in to.
    arl_args = set_single_dir(arl_args.mode + '_' + arl_args.render_type, arl_args)

    # Load an expert
    print('>> Copy expert from: {}'.format(arl_args.expert_location))
    shutil.copy(arl_args.expert_location + '/expert.pt', arl_args.log_dir + '/expert.pt')

    env = EnvWrapper(gym.make(arl_args.env_name), arl_args)
    observation_space = env.my_render()

    # Make the trainer class.
    trainer = A2Dagger.AdpAsymDagger(arl_args, img_stack=arl_args.frame_stack, img_size=arl_args.resize)

    # Grab an expert.
    expert_pol, expert_val = get_expert_network(trainer.params, trainer.env,
                                                dir_save_to=arl_args.expert_location,
                                                file_name='/expert.pt',
                                                create_new_expert=False)  # Load...

    # Mark the value function as asymmetric.
    learner_val = ValueDenseNet(env.state.shape[0])
    learner_val.ASYMMETRIC = True

    # Run the trainer!
    if trainer.params.mode == 'ete':
        # Define the policy with an encoder targeting state.
        learner_pol = DiscPolicy(observation_space.shape, env.action_space.shape[0], _encoder_dim=env.state.shape[0])

        # Inscribe the networks in to the class.
        trainer.define_networks(expert_pol, expert_val, learner_pol, learner_val)

        # Do the training.
        trainer.PretrainedEncoder()
    else:

        # Define the policy.
        learner_pol = DiscPolicy(observation_space.shape, env.action_space.shape[0])

        # Inscribe the networks in to the class.
        trainer.define_networks(expert_pol, expert_val, learner_pol, learner_val)

        # NOTE - these are not used, they can be safely deleted for verifcation of that fact.
        # They can be retained because it allows one to test the divergence of expert and agent.
        delattr(trainer, 'expert_pol')
        delattr(trainer, 'expert_val')

        trainer.logger = A2Dagger.init_logger(trainer.params)
        trainer.initial_log()
        trainer.dispatch_train()
        trainer.final_log()
