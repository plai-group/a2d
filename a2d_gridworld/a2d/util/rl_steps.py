# This code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Licence.  The full licence and information is available at:
# https://github.com/plai-group/a2d/blob/master/LICENCE.md
# © Andrew Warrington, J. Wilder Lavington, Adam Scibior, Mark Schmidt & Frank Wood.
# Accompanies the paper: Robust Asymmetric Learning in POMDPs, ICML 2021.

from torch.nn.utils import parameters_to_vector as flatten
from a2d.util.torch_utils import *


APPLICATION_BATCH_SIZE = 512


def adv_normalize(adv):
    """
    Normalise advantages to be unit Gaussian.
    :param adv:
    :return:
    """
    try:
        std = adv.std()
    except Exception as err:
        print(err)
        print(adv)
        raise RuntimeError("hmm, numerical error in adv_normalize.")
    assert std != 0. and not ch.isnan(std), 'Need nonzero std'
    n_advs = (adv - adv.mean())/(adv.std() + 1e-3)  # Increased stabilization...
    return n_advs


def surrogate_reward(params, adv_in, log_ps_new, log_ps_old, pds, clip_eps=None):
    """
    Compute a surrogate reward
    :param adv:
    :param log_ps_new:
    :param log_ps_old:
    :param clip_eps:
    :return:
    """

    # Sanity check making sure everything is the same size.
    assert shape_equal_cmp(log_ps_new, log_ps_old, adv_in)

    # Add entropy regularization if we are directly regularising the advantages.
    if (params.entropy_reg > 0.0) and params.regularize_adv:
        _entropy = - (pds.log() * pds).sum(-1)
        adv = adv_in + params.entropy_reg * _entropy
    else:
        adv = adv_in

    # Normalize the advantages.
    n_advs = adv_normalize(adv)

    # Ratio of new probabilities to old ones.
    ratio_new_old = ch.exp(log_ps_new - log_ps_old)

    # Weight by ratio.
    weighted_advantage = ratio_new_old * n_advs

    # Compute the entropy regularisation terms, if we did not directly regularise the entropy.
    if (params.entropy_reg > 0.0) and not params.regularize_adv:
        _entropy = - (pds.log() * pds).sum(-1)
        surr_rew = weighted_advantage + (params.entropy_reg * _entropy).mean()
    else:
        surr_rew = weighted_advantage

    return surr_rew


def value_loss_gae(vs, _, advantages, not_dones, params, old_vs, store=None):
    """
    Estimator for the value function that uses gae-smoothed advantage estimates a stabilized.
    :param vs:
    :param _:
    :param advantages:
    :param not_dones:
    :param params:
    :param old_vs:
    :param store:
    :param re:
    :return:
    """
    # Sanity check to make sure everything is the right dimension.
    assert shape_equal_cmp(vs, not_dones, old_vs, advantages)
    assert len(vs.shape) == 1

    # GAE-smoothed ahead value.
    val_targ = (old_vs + advantages).detach()

    # Apply some clipping to stabilize the estimate.
    vs_clipped = old_vs + ch.clamp(vs - old_vs, -params.clip_eps, params.clip_eps)

    # Find the ends of the trac
    sel = not_dones.bool()
    val_loss_mat_unclipped = (vs - val_targ)[sel]
    val_loss_mat_clipped = (vs_clipped - val_targ)[sel]

    # Presumably the inspiration for this is similar to PPO.
    if params.value_clipping:
        val_loss_mat = ch.max(val_loss_mat_unclipped, val_loss_mat_clipped)
    else:
        val_loss_mat = val_loss_mat_unclipped

    mse = val_loss_mat.pow(2).mean()

    return mse


def value_loss_returns(vs, returns, advantages, not_dones, params, old_vs, store=None):
    """
    Use a much simpler estimator that simply targets the reward ahead
    :param vs:
    :param returns:
    :param advantages:
    :param not_dones:
    :param params:
    :param old_vs:
    :param store:
    :return:
    """
    # Sanity check making sure everything is the right shape and type.
    assert shape_equal_cmp(vs, returns)
    not_dones = not_dones.bool()

    # Compute the values for those states that are not terminals.
    val_loss_mat = (vs - returns)[not_dones]

    # Compute the squared error.
    mse = val_loss_mat.pow(2).mean()

    return mse


def q_step(A2D_class, obs, target, actions, next_obs, q_opt, q_net, num_actions, params):
    """
    Copied from OSC.
    :param A2D_class:
    :param obs:
    :param target:
    :param actions:
    :param next_obs:
    :param q_opt:
    :param q_net:
    :param num_actions:
    :param params:
    :return:
    """

    # Helper function to generate minibatches.
    def sel(*args):
        return [v[selected].to(next(q_net.parameters()).device) for v in args]

    # Do this on the device.
    dev = params.device
    A2D_class.to_device()
    q_net.to(params.device)

    # Create minibatches indices.
    losses = []
    state_indices = np.arange(target.nelement())

    # Minibatch SGD
    for _ in range(params.q_epochs):

        # Re-shuffle minibatches.
        np.random.shuffle(state_indices)
        splits = np.array_split(state_indices, params.num_minibatches)

        # Iterate over minibatches.
        for selected in splits:

            # Zero the optimizer.
            q_opt.zero_grad()

            # Grab the minibatch.
            sel_obs, target_q, sel_actions = sel(obs, target.squeeze(), actions)

            # Format the one-hot action vector.
            sel_actions_ = ch.zeros((len(sel_obs), num_actions)).to(dev)
            sel_actions_ = sel_actions_.scatter(1, sel_actions.long().to(dev).unsqueeze(-1), 1).type(ch.double)
            sel_actions = sel_actions_.clone().to(sel_actions.device)

            # Append the action on to the flat observation vector.
            # TODO - this implementation only currently runs for flat state vectors.
            inputs = ch.cat((sel_obs.squeeze(), sel_actions.type(sel_obs.type())), dim=1).to(dev)
            pred_q = q_net.get_value(inputs).squeeze()

            # Compute the error and do backprop.
            loss = (pred_q - target_q).pow(2).mean()
            loss.backward()
            q_opt.step()

            # Track the loss.
            losses.append(loss.detach().to('cpu').item())

    # Send everything back to the CPU.
    q_net.to('cpu')
    A2D_class.to_cpu()


def value_step(A2D_class, all_states, returns, advantages, not_dones, net, val_opt, old_vs=None, opt_step=None,
               should_tqdm=False, params_in=None):
    """

    :param A2D_class:
    :param all_states:
    :param returns:
    :param advantages:
    :param not_dones:
    :param net:
    :param val_opt:
    :param old_vs:
    :param opt_step:
    :param should_tqdm:
    :param params_in:
    :return:
    """

    # Define a short helper function to generate the minibatch.
    def sel(*args):
        return [(v[selected].to(params.device) if v is not None else None) for v in args]

    # We can do this step on the gpu.
    if A2D_class is not None:
        params = A2D_class.params
        device = params.device
        A2D_class.to_device()
    else:
        params = params_in
        device = params.device
        net.to(device)

    # Get the right value function loss to use.
    if params.value_calc == 'gae':
        vf = value_loss_gae
    else:
        vf = value_loss_returns

    # Apply the value function being learned to each state.
    assert old_vs is None, "Depreciated old_vs."
    with ch.no_grad():

        values = []
        n_ex = len(all_states)
        idx = np.arange(0, n_ex, APPLICATION_BATCH_SIZE).astype(np.int)  # Set the size of the minibatch used.
        idx = np.concatenate((idx, [n_ex]))
        for _idx in range(len(idx[:-1])):
            values.append(net.get_value(all_states[idx[_idx]:idx[_idx + 1], ].to(device)))

        old_vs = ch.cat(values, dim=0).detach().to('cpu').squeeze()

    # Quick sanity check to make sure everything is the same dimensions.
    assert shape_equal_cmp(returns, advantages, not_dones, old_vs)

    # Apply a number of value function update epochs.
    for i in range(params.val_epochs):

        # Create minibatches from the data.
        state_indices = np.arange(returns.nelement())
        np.random.shuffle(state_indices)
        splits = np.array_split(state_indices, params.num_minibatches)

        # Minibatch SGD
        for selected in splits:
            val_opt.zero_grad()

            # Grab the minibatch
            sel_rets, sel_advs, sel_not_dones, sel_ovs, sel_states = \
                sel(returns, advantages, not_dones, old_vs, all_states)

            # Apply the value function
            vs = net.get_value(sel_states.to(device)).squeeze()

            # Another sanity check to make sure sizes are equal.
            assert shape_equal_cmp(vs, selected)

            # Take the value function update step.
            val_loss = vf(vs, sel_rets, sel_advs, sel_not_dones, params, sel_ovs)
            val_loss.backward()
            val_opt.step()


    # Send everything back to cpu.
    if A2D_class is not None: A2D_class.to_cpu()
    net.to('cpu')

    # Return the final value loss just for good measure.
    return val_loss.detach().to('cpu')


def _apply(net, params, _states, _actions):
    """

    :param net:
    :param params:
    :param _states:
    :param _actions:
    :return:
    """

    _pds, _action_log_probs = [], []

    # Work out the minibatches we are going to use.
    n_ex = len(_states)
    idx = np.arange(0, n_ex, APPLICATION_BATCH_SIZE).astype(np.int)
    idx = np.concatenate((idx, [n_ex]))

    # Iterate over the batches.
    for _idx in range(len(idx[:-1])):

        # Apply the network.
        _suff_stats = net(_states[idx[_idx]:idx[_idx + 1], ].to(params.device))

        # Add the action distribution to the store.
        _pds.append(_suff_stats.to('cpu'))

        # Get the log likelihood of the actions.
        _action_log_probs.append(net.get_loglikelihood(_suff_stats.to(net.device),
                                                       _actions[idx[_idx]:idx[_idx + 1], ].to(net.device)).to('cpu'))

    # Convert the lists into tensors.
    _pds = ch.cat(_pds, dim=0).squeeze().to('cpu')
    _action_log_probs = ch.cat(_action_log_probs, dim=0).squeeze().to('cpu')

    # Also output the actual probabilities.
    if type(_pds) == tuple:
        _pds = _pds[0]
    _pds = ch.softmax(_pds, dim=-1)

    return _pds, _action_log_probs


def trpo_step(all_states, actions, old_log_ps, rewards, returns, not_dones, advs, net, params, opt_step):
    """

    :param all_states:
    :param actions:
    :param old_log_ps:
    :param rewards:
    :param returns:
    :param not_dones:
    :param advs:
    :param net:
    :param params:
    :param opt_step:
    :return:
    """


    def _get_diff_params():
        """
        Grab all of the parameters that require gradients.
        :return:
        """
        __params = []
        for __p in net.parameters():
            if __p.requires_grad:
                __params.append(__p)
            else:
                pass
        return __params

    def fisher_product(x, damp_coef=1.):
        """
        Compute the fisher product.
        :param x:
        :param damp_coef:
        :return:
        """
        contig_flat = lambda q: ch.cat([y.contiguous().view(-1) for y in q])
        z = g @ x
        hv = ch.autograd.grad(z, _get_diff_params(), retain_graph=True)
        return contig_flat(hv).detach() + x*params.damping * damp_coef


    # This is a policy step, so ensure that the value function is not modified.
    if hasattr(net, 'vf'):
        for __p in net.vf.parameters():
            __p.requires_grad = False

    # If the encoder is pretrained, we dont want to change its parameters.
    if hasattr(net, 'encoder'):
        if net.encoder.PRETRAINED:
            assert next(net.encoder.parameters()).requires_grad == False, \
                'If the encoder is pretrained, its parameters must be fixed.'

    # Get the initial values of the parameters that we are going to be learning.
    initial_parameters = flatten(_get_diff_params()).clone()

    # Apply the network.
    pds, action_log_probs = _apply(net, params, all_states, actions)

    # Calculate the surrogate reward.
    surr_rew = surrogate_reward(params, advs, action_log_probs, old_log_ps, pds).mean()

    # Get the grad of the reward.
    grad = ch.autograd.grad(surr_rew, _get_diff_params(), retain_graph=True)
    flat_grad = flatten(grad)

    # Make fisher product estimator.
    num_samples = int(all_states.shape[0] * params.fisher_frac_samples)
    selected = np.random.choice(range(all_states.shape[0]), num_samples, replace=False)

    detached_selected_pds = select_prob_dists(pds, selected, detach=True)
    selected_pds = select_prob_dists(pds, selected, detach=False)

    kl = net.calc_kl(detached_selected_pds, selected_pds).mean()
    g = ch.autograd.grad(kl, _get_diff_params(), create_graph=True)
    g = flatten(g)

    # Find KL constrained gradient step.
    step = cg_solve(fisher_product, flat_grad, params.cg_steps).type(next(net.parameters()).type())
    max_step_coeff = (2 * params.max_kl / (step @ fisher_product(step))) ** 0.5
    max_trpo_step = max_step_coeff * step

    # Sometimes this step is ill-posed.  If it returns NaNs, increasing the
    # damping, and then try again.  If this doesn't work, then return nothing
    # as a failed update.
    if ch.any(ch.isnan(max_trpo_step)):
        params.damping *= 2.0
        step = cg_solve(fisher_product, flat_grad, params.cg_steps).type(next(net.parameters()).type())
        max_step_coeff = (2 * params.max_kl / (step @ fisher_product(step))) ** 0.5
        max_trpo_step = max_step_coeff * step

        if ch.any(ch.isnan(max_trpo_step)):
            print('Gradient step is ill-posed.  Increasing damping did not help.  Not taking step.')

            # Reset the value function gradient flag.
            if hasattr(net, 'vf'):
                for __p in net.vf.parameters():
                    __p.requires_grad = True

            # Send everything back to the CPU.
            net.to('cpu')

            return ch.tensor(0.0)
        else:
            # Increasing the damping worked.  Continue as before.
            print('Gradient step was ill-posed.  Temporarily increased damping.')

    # Backtracking line search.
    with ch.no_grad():

        # Backtracking function
        def backtrack_fn(s):
            assign(initial_parameters + s.data, net.parameters())

            net.to(params.device)
            test_pds, test_action_log_probs = _apply(net, params, all_states, actions)
            net.to('cpu')

            new_reward = surrogate_reward(params, advs.to('cpu'), test_action_log_probs.to('cpu'), old_log_ps.to('cpu'), test_pds).mean()

            # If there is no valid improvement, return a 'failure'.
            if new_reward <= surr_rew or net.calc_kl(pds, test_pds).mean() > params.max_kl:
                return -float('inf')

            # Otherwise return the improvement.
            return new_reward - surr_rew

        # Do the backtracking line search.
        expected_improve = flat_grad @ max_trpo_step
        final_step = backtracking_line_search(backtrack_fn, max_trpo_step, expected_improve, num_tries=params.max_backtrack)

        # Update the parameters.
        # Was net.parameters(), but these are constructed from the same iterator.
        assign(initial_parameters + final_step, _get_diff_params())

    # Reset the value function gradient flag.
    if hasattr(net, 'vf'):
        for __p in net.vf.parameters():
            __p.requires_grad = True

    # Send everything back to the CPU.
    net.to('cpu')

    return surr_rew

