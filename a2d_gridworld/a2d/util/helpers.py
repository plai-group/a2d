# This code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Licence.  The full licence and information is available at:
# https://github.com/plai-group/a2d/blob/master/LICENCE.md
# © Andrew Warrington, J. Wilder Lavington, Adam Scibior, Mark Schmidt & Frank Wood.
# Accompanies the paper: Robust Asymmetric Learning in POMDPs, ICML 2021.

import sys
from timeit import default_timer as dt
from a2d.util.torch_utils import *

st = {}


def discrete_kl_loss(learner_log_dist, expert_log_dist):
    """
    AW - Compute KL divergence.
    :param learner_log_dist:    (tensor):     log probability distribution over actions from learner.
    :param expert_log_dist:     (tensor):     log probability distribution over actions from expert.
    :return:
    """

    # Check type.
    if type(learner_log_dist) == tuple:
        learner_log_dist = learner_log_dist[0]
    if type(expert_log_dist) == tuple:
        expert_log_dist = expert_log_dist[0]

    # KL(p||q).
    loss = ch.sum(ch.softmax(expert_log_dist, dim=-1) * (expert_log_dist - learner_log_dist), dim=1)
    return loss.mean()


def advantage_and_return(params, rewards, values, not_dones):
    """
    AW - compute the discounted advantage and return as per GAE.
    :param self:
    :param rewards:
    :param values:
    :param not_dones:
    :return:
    """
    assert shape_equal_cmp(rewards, values, not_dones)

    V_s_tp1 = ch.cat([values[:, 1:], values[:, -1:]], 1) * (not_dones == 1)
    deltas = rewards + params.gamma * V_s_tp1 - values

    advantages = ch.zeros_like(rewards)
    returns = ch.zeros_like(rewards)
    indices = get_path_indices(not_dones)
    for agent, start, end in indices:
        advantages[agent, start:end] = discount_path(deltas[agent, start:end], params.lambda_ * params.gamma)
        returns[agent, start:end] = discount_path(rewards[agent, start:end], params.gamma)

    return advantages.clone().detach(), returns.clone().detach()


def do_log_header(results):
    """
    AW - Write the header for the csv file.  Needs to match the signature below in do_log(..).
    :param results:
    :return:
    """
    results.write('iters, timesteps, mdp_mean_reward_eval, pomdp_mean_reward_eval, mixed_mean_reward_eval, beta, '
                  'projection_loss, mdp_mean_reward_stoc, pomdp_mean_reward_stoc, mixed_mean_reward_stoc \n')
    results.flush()


def do_log(results, iters, timesteps, mdp_mean_reward_eval, pomdp_mean_reward_eval=0.0, mixed_mean_reward_eval=0.0, beta=1.0,
           projection_loss=0.0, mdp_mean_reward_stoc=0.0, pomdp_mean_reward_stoc=0.0, mixed_mean_reward_stoc=0.0):
    """
    AW - write signature must match the header above in do_log_header(..).
    :param results:
    :param iters:
    :param timesteps:
    :param mdp_mean_reward_eval:
    :param pomdp_mean_reward_eval:
    :param mixed_mean_reward_eval:
    :param beta:
    :param projection_loss:
    :param mdp_mean_reward_stoc:
    :param pomdp_mean_reward_stoc:
    :param mixed_mean_reward_stoc:
    :return:
    """
    if results is not None:
        # Pad with zeros to match the log shape of A2D.
        info = ",".join(("{}".format(_a) for _a in (iters, timesteps, mdp_mean_reward_eval, pomdp_mean_reward_eval,
                                                    mixed_mean_reward_eval, beta, projection_loss, mdp_mean_reward_stoc,
                                                    pomdp_mean_reward_stoc, mixed_mean_reward_stoc)))
        results.write(info + '\n')
        results.flush()


class Tee(object):
    """
    AW - Class for echoing output to file automatically.
    """
    def __init__(self, name):
        self.file = name
        self.stdout = sys.stdout
        sys.stdout = self
        self.tag = 'NoneSet'

    def __del__(self):
        sys.stdout = self.stdout

    def write(self, data):
        try:
            if data == '\n':
                _str = '\n'
                with open(self.file, 'a+') as f:
                    f.write(_str)
                self.stdout.write(_str)
                self.stdout.flush()
                return None

            if len(data) > 0:
                while data[0] == '\n':
                    data = data[1:]
                    self.write('\n')
                    if len(data) == 0: return None
            _str = ('[{:15.15}]: '.format(self.tag)) + str(data)

            with open(self.file, 'a+') as f:
                f.write(_str)
            self.stdout.write(_str)
            self.stdout.flush()

            p = 0
        except:
            pass


def start(_k):
    global st
    st[_k] = dt()


def end(_k):
    global st
    print('Timed {}: {: >05.2f}.'.format(_k, dt() - st[_k]))
