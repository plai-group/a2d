# This code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Licence.  The full licence and information is available at:
# https://github.com/plai-group/a2d/blob/master/LICENCE.md
# © Andrew Warrington, J. Wilder Lavington, Adam Scibior, Mark Schmidt & Frank Wood.
# Accompanies the paper: Robust Asymmetric Learning in POMDPs, ICML 2021.

# general imports
import torch
import numpy as np

import a2d.util.rl_steps as TRPO_STEPS

from torch.autograd import Variable
from a2d.util.torch_utils import *
from a2d.util.helpers import advantage_and_return

# We will batch applications on the GPU.  This is the default size.
# We could do this adaptively, but this seems simpler.
APPLICATION_BATCH_SIZE = 512


def importance_weight(A2D_class, policy_tag, states, obs, actions):
    """
    Same as OSC
    :param A2D_class:
    :param policy_tag:
    :param states:
    :param obs:
    :param actions:
    :return:
    """
    # Need to importance weight the advantages to get the gradient.
    assert 'A2D' == A2D_class.params.mode, 'Only IW in A2D'
    assert 'expert' == policy_tag, 'Only need to IW in A2D update to MDP policy'

    with torch.no_grad():

        A2D_class.to_device()

        # Compute numerator in IW.
        num = batched_evaluation(A2D_class.params, A2D_class.expert_pol, states.squeeze(), actions).exp().detach()

        # Compute denominator.
        den_s = num.detach().clone()
        den_o = batched_evaluation(A2D_class.params, A2D_class.learner_pol, obs, actions).exp().detach()
        den = (A2D_class.beta * den_s) + ((1 - A2D_class.beta) * den_o)

        # Multiply the advantages by the weight.
        eps = 1e-12
        weight = (num / (den + eps)).squeeze()

    A2D_class.to_cpu()
    return weight


def _batched_value_application(A2D_class, states, obs):
    """
    AW - evaluate the (mixture) value function at each sampled point in state space.
    :param A2D_class:   Trainer class.
    :param states:      States (omniscient) at which to evaluate vf.
    :param obs:         Observations (partial) at which to evaluate vf.
    :return:
    """

    # Create minibatches.
    values = []
    n_ex = len(states)
    idx = np.arange(0, n_ex, APPLICATION_BATCH_SIZE).astype(np.int)
    idx = np.concatenate((idx, [n_ex]))

    # Loop over the minibatches.
    for _idx in range(len(idx[:-1])):
        _v = A2D_class.get_mixture_value(states[idx[_idx]:idx[_idx + 1], ].to(A2D_class.params.device),
                                         obs[idx[_idx]:idx[_idx + 1], ].to(A2D_class.params.device))
        values.append(_v.detach().to('cpu'))

    values = ch.cat(values, dim=0).detach().to('cpu').squeeze(-1)
    return values


def batched_evaluation(p, policy_net, states, actions):
    """
    AW - Evalaute the log probability of the action each sampled point.
    :param p:           params.
    :param policy_net:  policy net to evaluate using.
    :param states:      sampled states to evaluate at.
    :param actions:     actions to evaluate the log prob of.
    :return:
    """

    # Create minibatches.
    fixed_log_prob = []
    n_ex = len(states)
    idx = np.arange(0, n_ex, APPLICATION_BATCH_SIZE).astype(np.int)  # Set the size of the minibatch used.
    idx = np.concatenate((idx, [n_ex]))

    # Never differentiate through this bit.
    with torch.no_grad():

        # Need to batch the application on the GPU.
        for _idx in range(len(idx[:-1])):

            # get the starting log prob for the loss
            action_probs = policy_net(Variable(states[idx[_idx]:idx[_idx+1]]).to(p.device))

            # Quickly do some type checking.
            if type(action_probs) == tuple:
                action_probs = tuple((_a.to('cpu') if _a is not None else None) for _a in action_probs)
            else:
                action_probs = action_probs.to('cpu')

            # Evaluate the log prob of the actions.
            fixed_log_prob.append(policy_net.get_loglikelihood(action_probs, actions[idx[_idx]:idx[_idx+1]].to('cpu')))

    # Make vector and return.
    fixed_log_prob = torch.cat(fixed_log_prob, dim=0).detach().to('cpu')
    return fixed_log_prob


def q(A2D_class, states, obs, actions):
    """
    AW - evaluate the (mixture) advantage using the Q function
    (which is basically a value function with the actions also input).

    NOTE - this automatically subtracts off the value function as a baseline
    since we only ever use the Q in this way.

    :param A2D_class:   Trainer class.
    :param states:      sampled states to evaluate at.
    :param obs:         sampled observations to evaluate at.
    :param actions:     samples actions to evaluate at.
    :return:
    """
    # We never differentiate through the Q function.
    with torch.no_grad():

        # Move to  GPU.
        A2D_class.to_device()

        # Format the tensor of actions into a one-hot tensor.
        _sel_actions = ch.zeros((len(obs), A2D_class.learner_pol.num_outputs)).to(actions.device)
        _sel_actions = _sel_actions.scatter(1, actions.long().unsqueeze(-1), 1).type(ch.double)

        # Get the mixture q value.
        _input = ch.cat((obs.squeeze().to(A2D_class.params.device), _sel_actions.to(A2D_class.params.device)), dim=1)
        _q_learner = A2D_class.learner_q.get_value(_input).squeeze()
        _input = ch.cat((states.squeeze().to(A2D_class.params.device), _sel_actions.to(A2D_class.params.device)), dim=1)
        _q_expert = A2D_class.expert_q.get_value(_input).squeeze()
        _q = (A2D_class.beta * _q_expert) + ((1 - A2D_class.beta) * _q_learner)

        # NOTE - comment this to remove VF.
        # _v = 0.0
        _v = A2D_class.get_mixture_value(states.to(A2D_class.params.device),
                                         obs.to(A2D_class.params.device)).squeeze()

        # Compute the advantages.
        advantages = (_q - _v).detach().to('cpu')
        A2D_class.to_cpu()

    return advantages


def trpo_step(A2D_class, batch, policy_tag):
    """
    AW - Take a RL step.
    :param A2D_class:   Trainer class.  Holds all params and networks etc.
    :param batch:       Samples trajectories/rollouts to use.
    :param policy_tag:  String indicating which networks we are updating this step.
    :return:            None
    """

    # Format the arguments.
    A2D_class.to_cpu()
    states = (torch.tensor(batch.state))
    obs = torch.stack(batch.stored_observe)
    rewards = (torch.tensor(batch.reward))
    not_dones = (torch.tensor(batch.mask) == 1)
    actions = torch.tensor(np.asarray(batch.action))

    # Sort out which networks are updated where and with what variables.
    if 'expert' == policy_tag:
        # Updating the expert policy and value fn.
        fn_pol = A2D_class.expert_pol
        fn_val = A2D_class.expert_val
        fn_q = A2D_class.expert_q
        fn_val_opt = A2D_class.optims['expert_val']
        fn_q_opt = A2D_class.optims['expert_q']
        policy_step = True
        inputs = states

        if A2D_class.params.mode == 'A2D':
            do_importance_weight = True
        else:
            do_importance_weight = False

    elif policy_tag == 'learner_val':
        # Updating the learning value fn only.  Used in A2D, as the policy
        # is updated via the projection step.
        fn_pol = A2D_class.learner_pol
        fn_val = A2D_class.learner_val
        fn_q = A2D_class.learner_q
        fn_val_opt = A2D_class.optims['learner_val']
        fn_q_opt = A2D_class.optims['learner_q']
        policy_step = False
        inputs = obs
        do_importance_weight = False

    elif policy_tag == 'learner':
        # Update the entire learner.
        fn_pol = A2D_class.learner_pol
        fn_val = A2D_class.learner_val
        fn_q = A2D_class.learner_q
        fn_val_opt = A2D_class.optims['learner_val']
        fn_q_opt = A2D_class.optims['learner_q']
        policy_step = True
        inputs = obs
        do_importance_weight = False

    else:
        raise RuntimeError('Unrecognised tag.')

    # Compute the action logs probs, advantages, returns etc.
    with torch.no_grad():
        action_log_probs = batched_evaluation(A2D_class.params, fn_pol, inputs, actions)  
        
        # Get the mixture of VFs & Compute advatnages and returns.
        A2D_class.to_device()
        values = _batched_value_application(A2D_class, states, obs)
        advantages, returns = advantage_and_return(A2D_class.params, rewards.clone().unsqueeze(0), values.unsqueeze(0), not_dones.unsqueeze(0))
        A2D_class.to_cpu()

        # Make sure everything is the right size...
        advantages = advantages.squeeze()
        returns = returns.squeeze()
        values = values.squeeze()
        assert shape_equal_cmp(advantages, returns, values)

    # Take value function steps.
    if not fn_val.ASYMMETRIC:
        TRPO_STEPS.value_step(A2D_class, inputs, returns, advantages, not_dones, fn_val, fn_val_opt)
    else:
        TRPO_STEPS.value_step(A2D_class, states, returns, advantages, not_dones, fn_val, fn_val_opt)

    # Updating the q function here.
    if A2D_class.params.USE_Q:
        TRPO_STEPS.q_step(A2D_class, inputs, returns.squeeze(), actions, None, fn_q_opt, fn_q, fn_pol.num_actions, A2D_class.params)

    # Take policy steps.  Replaced unused args with None.
    if policy_step:

        # Are we importance weighting the gradients, as per A2D?
        if do_importance_weight:

            # If we are using the Q function, then explicitly replace the computed advantages.
            if A2D_class.params.USE_Q:
                advantages = q(A2D_class, states, obs, actions)

            # Do the importance weight.
            advantages *= importance_weight(A2D_class, policy_tag, states, obs, actions)
            assert torch.all(torch.isfinite(advantages)), 'Advantages must be finite...'

        # Take the RL step.
        TRPO_STEPS.trpo_step(inputs, actions, action_log_probs, None, None, None, advantages, fn_pol, A2D_class.params, None)


    # NOTE - disabled annealing.
    # if policy_step:
    #     # Anneal the KL in the TRPO.
    #     A2D_class.params.max_kl *= A2D_class.params.kl_factor
    #     # A2D_class.params.max_kl += A2D_class.params.kl_increment

