# This code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Licence.  The full licence and information is available at:
# https://github.com/plai-group/a2d/blob/master/LICENCE.md
# © Andrew Warrington, J. Wilder Lavington, Adam Scibior, Mark Schmidt & Frank Wood.
# Accompanies the paper: Robust Asymmetric Learning in POMDPs, ICML 2021.

# general imports
import sys
import os
import time
import gym
import torch
import timeit

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from datetime import datetime
from copy import deepcopy
from torch.distributions.bernoulli import Bernoulli
from a2d.models.models import DiscPolicy, ValueDenseNet
from a2d.util.sampler import sample_batch
from a2d.environments.env_wrap import EnvWrapper
from a2d.environments.env_manager import EnvManager
from a2d.util.inits import init_replay_buffer, init_logger
from a2d.util.helpers import discrete_kl_loss, do_log


class A2D_base(object):
    """
    Base class to implement most stock functionality.
    """

    def __init__(self, args, img_stack=4, img_size=84, wrappers=None):
        
        self.params = args  # set args for later.
        self.img_stack = img_stack  # set image stack.
        self.img_size = img_size  # Obs size.
        self.buffer_size = args.buffer_size
        self.max_proj_iter = args.max_proj_iter  # set projections per iteration.
        self.device = args.device  # Device..
        
        # Initialise some holders that will be used later.
        self.logger = None                                                      # init logging.
        self.expert_pol, self.expert_val, self.expert_q = None, None, None      # init expert networks.
        self.learner_pol, self.learner_val, self.learner_q = None, None, None   # init learner networks.
        self.mixture_reward_per_iter_det = []                                   # something to store the reward per episode.
        self.expert_reward_per_iter_det = []                                    # something to store the reward per episode.
        self.learner_reward_per_iter_det = []                                   # something to store the reward per episode.
        self.mixture_reward_per_iter_sto = []                                   # something to store the reward per episode.
        self.expert_reward_per_iter = []                                        # something to store the reward per episode.
        self.learner_reward_per_iter = []                                       # something to store the reward per episode.
        self.last_projection_loss = 90210                                       # Just to track any projection losses.
        self.current_projection_loss = 90210                                    # Just to track any projection losses.

        # If we are doing straight DAgger, fix the return types.
        if args.mode == 'D':
            self.params.render_type = 'state'  # Symmetric so observe state.
            self.params.frame_stack = 1  # State -> markov -> one frame only.

        if args.render_type == 'state':
            self.params.frame_stack = 1  # State -> markov -> one frame only.

        self.env = self.env_constructor(wrappers)  # generate environments.

        self.beta = torch.tensor(1.)  # set dagger mixture param.
        self.beta_decay = args.beta_decay  # set dagger mixture decay param.

        self.replay_buffer = init_replay_buffer(args)  # set replay buffer.

        # Most of the time we don't use annealing, so it doesn't matter, but it is a good
        # thing to think about.
        self.params.kl_factor = (self.params.max_kl_final / self.params.max_kl) ** float(1.0 / self.params.iters)
        # self.params.kl_increment = (self.params.max_kl_final - self.params.max_kl) / float(self.params.iters)

        # This will track timings for various things.  They aren't 100% robust, but are handy indicators.
        self.timings = {'data': 0.0,
                        'rl_ex': 0.0,
                        'rl_le': 0.0,
                        'rl_le_val': 0.0,
                        'proj': 0.0,
                        'log': 0.0}
        

    def env_constructor(self, wrappers):
        """
        AW - Call the constructor for wrapping environments.  This doesn't 
        do a whole lot here, but basically allows us to wrap the environment
        so that we can define more useful attributes such as framestacks.
        :param wrappers:    Additional functions to wrap standard calls. Not used.
        :return:            EnvWrapper class object.
        """
        # Make gym environment.
        env = gym.make(self.params.env_name)
        
        # Now is there an additional wrapper we need to add.
        env = EnvManager(env, wrappers)

        # generate the main wrapper class to get correct attributes
        return EnvWrapper(env, params=self.params)


    def discrete_kl_loss(self, learner_log_dist, expert_log_dist):
        """
        AW - wrapper for computing the discrete KL divergence.
        :param learner_log_dist:    log probs of actions from learner.
        :param expert_log_dist:     log probs of actions from expert.
        :return:                    KL divergence.
        """
        return discrete_kl_loss(learner_log_dist, expert_log_dist)


    def to_cpu(self):
        """
        AW - send all of the networks to the CPU.
        :return:
        """
        if hasattr(self, 'expert_pol'): self.expert_pol = self.expert_pol.to('cpu')
        if hasattr(self, 'expert_val'): self.expert_val = self.expert_val.to('cpu')
        if hasattr(self, 'learner_pol'): self.learner_pol = self.learner_pol.to('cpu')
        if hasattr(self, 'learner_val'): self.learner_val = self.learner_val.to('cpu')

        if self.params.USE_Q:
            if hasattr(self, 'learner_q'): self.learner_q = self.learner_q.to('cpu')
            if hasattr(self, 'expert_q'): self.expert_q = self.expert_q.to('cpu')


    def to_device(self):
        """
        AW - send all of the networks to the selected device.
        :return:
        """
        if hasattr(self, 'expert_pol'): self.expert_pol = self.expert_pol.to(self.device)
        if hasattr(self, 'expert_val'): self.expert_val = self.expert_val.to(self.device)
        if hasattr(self, 'learner_pol'): self.learner_pol = self.learner_pol.to(self.device)
        if hasattr(self, 'learner_val'): self.learner_val = self.learner_val.to(self.device)

        if self.params.USE_Q:
            if hasattr(self, 'learner_q'): self.learner_q = self.learner_q.to(self.device)
            if hasattr(self, 'expert_q'): self.expert_q = self.expert_q.to(self.device)


    def expert_sampling_dist(self, state_, observe_, eval_=True, return_dist=False):
        """
        AW - Sample from the expert policy.
        :param state_:      Omniscient state vector. 
        :param observe_:    Observation available to the learner.  Not used here.
        :param eval_:       Return the argmax of the policy, or a random sample.
        :return:            Action, and 1, indicating this was a sample from the expert.
        """
        if return_dist:
            action, dist = self.expert_pol.sample_action(state_.unsqueeze(0), eval_, return_dist=True)
            return action, 1, dist
        else:
            action = self.expert_pol.sample_action(state_.unsqueeze(0), eval_, return_dist=False)
            return action, 1


    def learner_sampling_dist(self, state_, observe_, eval_=True, return_dist=False):
        """
        AW - Sample from the learner policy.
        :param state_:      Omniscient state vector.  Not used here.
        :param observe_:    Observation available to the learner.
        :param eval_:       Return the argmax of the policy, or a random sample.
        :param return_dist: Return the distribution over actions? only returns the first entry currently.
        :return:            Action, and 0, indicating this was a sample from the learner.
        """
        lps = self.learner_pol(observe_.unsqueeze(0))
        ps = lps.exp()

        if eval_:
            actions = torch.argmax(ps, dim=1)
        else:
            dist = torch.distributions.categorical.Categorical(ps)
            actions = dist.sample()

        if not return_dist:
            return actions, 0
        else:
            return actions, 0, ps[0]


    def mixed_sampling_dist(self, state_, observe_, eval_=True):
        """
        AW - sample from the \beta-mixture of policies.
        :param state_:      Omniscient state vector.  
        :param observe_:    Observation available to the learner.
        :param eval_:       Return the argmax of the policy, or a random sample.
        :return:            Action, and {0,1}, indicating if this was a sample from the expert.
        """
        # set mixture dist
        beta_i = Bernoulli(self.beta).sample()
        # sample action
        if not beta_i:
            return self.learner_sampling_dist(state_, observe_, eval_)
        else:
            return self.expert_sampling_dist(state_, observe_, eval_)


    def get_mixture_value(self, state, obs):
        """
        AW - Calculate the mixture of the value functions.
        :param state:
        :param obs:
        :return:
        """
        # Do some type checking.
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        if not torch.is_tensor(obs):
            obs = torch.stack(obs).squeeze()

        if hasattr(self, 'expert_val'):
            if self.beta > 0.0:
                mdp = self.expert_val(state.to(self.device))
            else:
                mdp = 0
        else:
            mdp = 0

        if hasattr(self, 'learner_val'):
            if self.learner_val.ASYMMETRIC:
                pom = self.learner_val.value(state.to(self.device))
            else:
                pom = self.learner_val.value(obs.to(self.device))
        else:
            pom = 0

        return (self.beta * mdp) + ((1 - self.beta) * pom)


    def pretrain(self, pre_train):
        """
        Sometimes we may want to do a pretraining step.  Most of the time we won't use
        this.  This simply calls step to generate some data, and then does a projection
        step to align the expert and agent from the outset.
        :param pre_train:  (bool):  do pretraining step.
        :return: None
        """
        # pre-train
        if pre_train and self.params.pretrain_batch_size > 0 and (not self.params.cotrain):
            print('Taking pre-training step ...')
            temp = deepcopy(self.params)
            self.params.batch_size = self.params.pretrain_batch_size
            self.params.max_proj_iter = self.params.pretraining_projection_steps
            self.step(-1, rl_update=False, sampling_dist='expert', verbose=True)
            self.params = deepcopy(temp)
            self.logging(0)  # For a logging step.
            print('Pretraining step complete')  # , Time: {}.'.format(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
        else:
            print('Skipping pretraining.')


    def define_networks(self, expert_pol, expert_val, learner_pol, learner_val):
        """
        AW - define any networks and optimizers we want to use.
        NOTE - not all of these optimizers will be used.  For instance, ``learner`` is
        learned using TRPO, but I find it quite handy to have access to an object that
        tracks gradients/parameters etc.  Just don't call step!
        :return:
        """
        # define networks
        self.expert_pol, self.expert_val = expert_pol, expert_val
        self.learner_pol, self.learner_val = learner_pol, learner_val

        # define optimizers
        self.optims = {}

        if self.learner_pol is not None:
            self.optims['learner_pol'] = optim.Adam(self.learner_pol.parameters(),
                                                    lr=self.params.lr, eps=self.params.eps)

        if self.learner_val is not None:
            self.optims['learner_val'] = optim.Adam(self.learner_val.parameters(),
                                                    lr=self.params.val_lr, eps=self.params.eps)

        if self.expert_pol is not None:
            self.optims['expert_pol'] = optim.Adam(self.expert_pol.parameters(),
                                                   lr=self.params.rl_lr, eps=self.params.eps)

        if self.expert_val is not None:
            self.optims['expert_val'] = optim.Adam(self.expert_val.parameters(),
                                                   lr=self.params.val_lr, eps=self.params.eps)

        if self.params.USE_Q:
            # Now also create the q network.
            self.expert_q = ValueDenseNet(self.expert_val.obs_shape + self.learner_pol.num_outputs, slim_init=False)
            self.optims['expert_q'] = optim.Adam(self.expert_q.parameters(),
                                                 lr=self.params.q_lr, eps=self.params.eps)

            # Now also create the q network.
            self.learner_q = ValueDenseNet(self.learner_pol.obs_shape[0] + self.learner_pol.num_outputs, slim_init=False)
            self.optims['learner_q'] = optim.Adam(self.learner_q.parameters(),
                                                  lr=self.params.q_lr, eps=self.params.eps)
        else:
            self.expert_q, self.learner_q = None, None
            self.optims['expert_q'], self.optims['learner_q'] = None, None


    def save_as_expert(self, reward):
        """
        If we are doing RL(MDP) / RL in the expert, then we need to save the policy
        in an appropriate format so it can be retrieved later as required.
        :param reward:  (float):    reward earned by network.
        :return:
        """
        # Try and save an intermediate version of the model.
        try:
            try:
                expert_dict = {'expert_network': self.expert_pol.state_dict(),
                               'expert_value_net': self.expert_val.state_dict(),
                               'expert_performance': reward,
                               'expert_render_type': self.params.render_type,
                               'expert_env_tag': self.params.env_tag,
                               'expert_env_name': self.params.env_name,
                               'params': {'dir': self.params.log_dir,
                                          'git_commit': self.params.git_commit,
                                          'machine': self.params.machine}}

                if reward > self.best_loss:
                    self.best_loss = deepcopy(reward)
                    torch.save(expert_dict, self.params.expert_location + '/expert.pt')
                    # print(self.best_loss)

            except Exception as err:
                self.best_loss = reward
                print('Saving expert into: {}: '.format(self.params.expert_location + '/expert.pt'))
                torch.save(expert_dict, self.params.expert_location + '/expert.pt')
                print('Successfully saved expert.')
        except Exception as err:
            pass


    def do_print(self, iter):
        """
        Print some stuff to the screen.
        :param iter:    (int):  Current iteration.
        :return: 
        """
        try:
            elapsed_time = deepcopy(timeit.default_timer() - self._last_log_time)
        except:
            elapsed_time = 0.0  # Probably on first log.
        self._last_log_time = timeit.default_timer()

        print('Ep: {: >03d}/{: >03d}, '.format(iter, self.params.iters) +
              'TBL: {: >06.1f}s, '.format(elapsed_time) +
              'Tr TBL: {: >06.1f}s, '.format(self._inter_log_train_time) +
              'B: {: 0.2f}, '.format(self.beta) +
              'Ints: {: >08d}, '.format(self.replay_buffer.samples_proccesed) +
              'Exp-d {: >07.2f}, '.format(self.expert_reward_per_iter_det[-1]) +
              'Lea-d {: >07.2f}, '.format(self.learner_reward_per_iter_det[-1]) +
              'Exp-s {: >07.2f}, '.format(self.expert_reward_per_iter[-1]) +
              'Lea-s {: >07.2f}, '.format(self.learner_reward_per_iter[-1]) +
              ('Div: {: >6.4f} '.format(self.current_projection_loss) if self.current_projection_loss != 90210.0 else 'Div: N/A ') +
              '| T: ' + ', '.join(['{:}: {: 5.0f}'.format(_k, self.timings[_k]) for _k in self.timings]).replace('\n', '')
              )


    def logging(self, iter, verbose=True, log_it=True):
        """
        Subroutine for regular logging.  This will evaluate the experts and learners performance
        for both stochastic and deterministic policies, if the current iteration is a logging
        iteration (i.e.  iter % self.params.log_interval == 0).  It will also evaluate the
        divergence between policies.  If doing RL(MDP), then it will also save the expert in the
        required format if the current policy is better than the previous policy.

        Basically a whole load of admin.  Grumble.

        :param iter:        (int):      iteration.
        :param verbose:     (bool):     more textual output?
        :param log_it:      (bool):     Default is true, but this allows logging to be forcibly
                                        avoided if set to false.  Logging can be expensive, and
                                        so feasibly you might want to supress logging.
        :return:    None, just a whole bunch of side effects.
        """
        if log_it and iter % self.params.log_interval == 0:
            _st = timeit.default_timer()

            try:
                self._inter_log_train_time = deepcopy(timeit.default_timer() - self._inter_log_train_time)
            except:
                self._inter_log_train_time = 0.0  # Probably on first log.

            # Define defaults.
            expert_reward_batch_det, mixed_reward_batch_det, learner_reward_batch_det = 0.0, 0.0, 0.0
            expert_reward_batch_sto, mixed_reward_batch_sto, learner_reward_batch_sto = 0.0, 0.0, 0.0

            self.to_device()

            # Copy the args, and define the (fixed) arguments for the evalautions.
            temp_args = deepcopy(self.params)
            temp_args.batch_size = temp_args.eval_batch_size


            # Generate some trajectory figures.
            if self.params.DO_TRAJECTORY_RENDER and (iter % 20 == 0):

                eval_ = True

                if self.params.render_type == 'state':
                    dist = self.expert_sampling_dist
                else:
                    dist = self.learner_sampling_dist

                _st = timeit.default_timer()
                temp_args_ = deepcopy(temp_args)
                temp_args_.batch_size = 100
                print('Making trajectory figures...')
                kwargs = {'env': self.env,
                          'args': temp_args_,
                          'eval_': eval_,
                          'store': False,
                          'full_trajectory': True,
                          'SAVE_RENDERS': True,
                          'ITER': iter}
                sample_batch(dist, **kwargs)
                print('... Done making trajectory figures. Time: {:5.2f}s.'.format(timeit.default_timer() - _st))


            # Deterministic rollouts.
            with torch.no_grad():

                kwargs = {'env': self.env,
                          'args': temp_args,
                          'eval_': True,
                          'store': False,
                          'full_trajectory': True}

                if not hasattr(self, 'expert_pol'):
                    expert_reward_batch_det = 0.0
                else:
                    expert_reward_batch_det = sample_batch(self.expert_sampling_dist, **kwargs)[0]

                if not hasattr(self, 'learner_pol'):
                    learner_reward_batch_det = 0.0
                else:
                    learner_reward_batch_det = sample_batch(self.learner_sampling_dist, **kwargs)[0]


            # Stochastic rollouts.
            with torch.no_grad():

                kwargs = {'env': self.env,
                          'args': temp_args,
                          'eval_': False,
                          'store': False,
                          'full_trajectory': True}

                if not hasattr(self, 'expert_pol'):
                    expert_reward_batch_sto = 0.0
                else:
                    expert_reward_batch_sto = sample_batch(self.expert_sampling_dist, **kwargs)[0]

                if not hasattr(self, 'learner_pol'):
                    learner_reward_batch_sto = 0.0
                else:
                    learner_reward_batch_sto = sample_batch(self.learner_sampling_dist, **kwargs)[0]


            # Evaluate the divergence between the MDP and POMDP.
            if hasattr(self, 'expert_pol') and hasattr(self, 'learner_pol'):
                try:
                    if len(self.replay_buffer) > 0:
                        self.current_projection_loss = self.mini_batch_update(-1, verbose=False,
                                                                              update=False, augmentate=False)
                        pass
                    else:

                        print('No samples to use.  Generating some test samples and then resetting buffer.  ')
                        _, _temp_batch, _ = sample_batch(self.expert_sampling_dist, self.env, self.params, eval_=False)
                        self.replay_buffer.push(deepcopy(_temp_batch), )
                        self.current_projection_loss = self.mini_batch_update(-1, verbose=False,
                                                                              update=False, augmentate=False)
                        self.replay_buffer.hard_reset()
                except:
                    pass

            # Append results.
            self.expert_reward_per_iter_det.append(expert_reward_batch_det)
            self.learner_reward_per_iter_det.append(learner_reward_batch_det)
            self.mixture_reward_per_iter_det.append(mixed_reward_batch_det)
            self.expert_reward_per_iter.append(expert_reward_batch_sto)
            self.learner_reward_per_iter.append(learner_reward_batch_sto)
            self.mixture_reward_per_iter_sto.append(mixed_reward_batch_sto)

            do_log(self.logger, iter, self.replay_buffer.samples_proccesed, expert_reward_batch_det,
                   learner_reward_batch_det, mixed_reward_batch_det, self.beta, self.current_projection_loss,
                   expert_reward_batch_sto, learner_reward_batch_sto, mixed_reward_batch_sto)

            # Save some information.
            torch.save(self._gen_info_dict(iter), os.path.join(self.params.log_dir, 'info_dict.pt'))

            # Reset the timer.
            self.timings['log'] += timeit.default_timer() - _st     # Increase the timer for logging.
            if verbose: self.do_print(iter)
            self._inter_log_train_time = timeit.default_timer()         # Probably on first log.
            self.to_cpu()

            # Dump the expert out as well.
            if self.params.save_as_expert:
                self.save_as_expert(expert_reward_batch_sto)


    def _gen_info_dict(self, iter):
        """
        Generate a dictionary of information at this iter to save.
        :param iter:    (int):  iteration.
        :return:        (dict): information dictionary.
        """
        self.info_dict = {'iter': iter,
                          'optims': self.optims,

                          'expert_val_class': str(type(self.expert_val)) if hasattr(self, 'expert_val') else None,
                          'expert_val_params': self.expert_val.state_dict() if hasattr(self, 'expert_val') else None,

                          'expert_pol_class': str(type(self.expert_pol)) if hasattr(self, 'expert_pol') else None,
                          'expert_pol_params': self.expert_pol.state_dict() if hasattr(self, 'expert_pol') else None,

                          'learner_val_class': str(type(self.learner_val)) if hasattr(self, 'learner_val') else None,
                          'learner_val_params': self.learner_val.state_dict() if hasattr(self, 'learner_val') else None,

                          'learner_pol_class': str(type(self.learner_pol)) if hasattr(self, 'learner_pol') else None,
                          'learner_pol_params': self.learner_pol.state_dict() if hasattr(self, 'learner_pol') else None,

                          'det_mixture_info': self.mixture_reward_per_iter_det,
                          'det_mdp_info': self.expert_reward_per_iter_det,
                          'det_pomdp_info': self.learner_reward_per_iter_det,
                          'sto_mixture_info': self.mixture_reward_per_iter_sto,
                          'sto_mdp_info': self.expert_reward_per_iter,
                          'sto_pomdp_info': self.learner_reward_per_iter,
                          'timings': self.timings}
        return self.info_dict


    def initial_log(self):
        """
        AW - print any stuff out before starting training, also run any
        initial things before pretraining / training.
        :return:
        """
        print('==================================================')
        print('Starting {} Training.'.format(self.params.mode))
        print('Start time: {}'.format(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))


        if hasattr(self, 'expert_val'):
            print('Expert Value function input shape:    ' +
                  str(self.expert_val.obs_shape))
        else:
            print("No Expert value function.")


        if hasattr(self, 'expert_pol'):
            print('Expert Policy function input shape:   ' +
                  str(self.expert_pol.obs_shape))
        else:
            print("No Expert policy.")


        if hasattr(self, 'learner_pol'):
            if 'Identity' in str(type(self.learner_pol.encoder)):
                print("Learner Policy encoder is identity function.")
            else:
                print('Learner Policy encoder function input shape:  ' +
                      str(self.learner_pol.encoder.obs_shape))
        else:
            print("No Learner policy.")


        if hasattr(self, 'learner_val'):
            if self.learner_val.ASYMMETRIC:
                print('Learner (asymmetric) Value function input shape:  ' +
                      str(self.learner_val.obs_shape))

            else:
                if 'Identity' in str(type(self.learner_val.encoder)):
                    print('Learner (symmetric) value is identity function.')
                else:
                    print('Learner (symmetric) value function input shape:  ' +
                          str(self.learner_val.encoder.obs_shape))
        else:
            print("No Learner value function.")

        print('Using lambda value: ' + str(self.params.lambda_))
        print('Using entropy reg:  ' + str(self.params.entropy_reg))

        print('==================================================')

        # Do initial logging step.
        if self.params.cluster:
            self.logging(0)
            pass
        else:
            self.logging(0)
            pass


    def final_log(self):
        """
        Clean up, run any final logging steps, save results, and close up shop.
        :return:
        """
        # also store a tensor
        self.info_dict = self._gen_info_dict(self.params.iters)

        # Print out the timigs.
        print('Timings: ' + ', '.join(['{:}: {: 6.0f}s'.format(_k, self.timings[_k]) for _k in self.timings]).replace('\n', ''))

        # Do one big final log...
        with torch.no_grad():
            self.to_device()
            temp_args = deepcopy(self.params)
            if temp_args.cluster:
                temp_args.batch_size *= 5  # Use a bigger batch for final evaluation.
            else:
                temp_args.batch_size = 2000

            kwargs = {'env': self.env,
                      'args': temp_args,
                      'eval_': True,
                      'store': False,
                      'full_trajectory': True}

            # Deterministic evaluations.
            kwargs['eval_'] = True
            expert_reward_batch_det = sample_batch(self.expert_sampling_dist, **kwargs)[0] if hasattr(self, 'expert_pol') else 0.0
            learner_reward_batch_det = sample_batch(self.learner_sampling_dist, **kwargs)[0] if hasattr(self, 'learner_pol') else 0.0

            # Stochastic evaluations.
            kwargs['eval_'] = False
            expert_reward_batch_sto = sample_batch(self.expert_sampling_dist, **kwargs)[0] if hasattr(self, 'expert_pol') else 0.0
            learner_reward_batch_sto = sample_batch(self.learner_sampling_dist, **kwargs)[0] if hasattr(self, 'learner_pol') else 0.0

            # Evalaute the divergence between the MDP and POMDP.
            if hasattr(self, 'expert_pol') and hasattr(self, 'learner_pol'):
                try: current_projection_loss = self.mini_batch_update(-1, verbose=False, update=False)
                except: current_projection_loss = 90210.0
            else:
                current_projection_loss = 90210.0

        self.info_dict['final_stats'] = {'mdp_det': expert_reward_batch_det,
                                         'mdp_sto': expert_reward_batch_sto,
                                         'pom_det': learner_reward_batch_det,
                                         'pom_sto': learner_reward_batch_sto,
                                         'divergence': current_projection_loss
                                         }

        print('\nFinal statistics: ' +
              'Interactions: {},  '.format(self.replay_buffer.samples_proccesed) +
              'MDP R (det): {: 6.2f},  '.format(expert_reward_batch_det) +
              'POMDP R (det): {: 6.2f},  '.format(learner_reward_batch_det) +
              'MDP R (sto): {: 6.2f},  '.format(expert_reward_batch_sto) +
              'POMDP R (sto): {: 6.2f},  '.format(learner_reward_batch_sto) +
              'Divergence: {: 6.2f},  '.format(current_projection_loss))

        # Do some final logging.
        if self.params.DO_TRAJECTORY_RENDER:
            print('Doing final trajectory rendering.')

            if self.params.render_type == 'state':
                dist = self.expert_sampling_dist
            else:
                dist = self.learner_sampling_dist

            _st = timeit.default_timer()
            temp_args.batch_size = 500
            kwargs = {'env': self.env,
                      'args': temp_args,
                      'eval_': False,
                      'store': False,
                      'full_trajectory': True,
                      'SAVE_RENDERS': True,
                      'ITER': 'final'}
            sample_batch(dist, **kwargs)
            print('Time for {} samples: {}s'.format(temp_args.batch_size, timeit.default_timer() - _st))

        torch.save(self.info_dict, os.path.join(self.params.log_dir, 'info_dict.pt'))   # save information.
        self.logger.close()                                                             # close logging file.


    def _train_encoder(self):
        """
        Train a fixed encoder using rollouts from under the MDP.  This encoder is learned in-place
        in the currently defined learner policy.
        :return: Encoder is returned as a side effect in the learner policy.
        """

        # Check to make sure we are using images.
        assert self.params.return_type == 'observe' or self.params.return_type == 'full_observe'
        if self.params.return_type == 'full_observe':
            print('\n\nWarning: using full observe for encoder.\n\n')

        # Generate some data under the expert.
        batch_size_temp = deepcopy(self.params.batch_size)
        self.params.batch_size = self.params.preenc_encoder_samples
        _, _preenc_batch, _ = sample_batch(self.expert_sampling_dist, self.env, self.params, eval_=True)
        self.replay_buffer.push(deepcopy(_preenc_batch), )
        self.params.batch_size = deepcopy(batch_size_temp)

        # Train the encoder.
        _preenc_opt = torch.optim.Adam(self.learner_pol.encoder.parameters(), lr=3e-4)
        _preenc_loss = torch.nn.MSELoss()
        self.learner_pol.encoder.to(self.params.device)

        # Do mini-batch steps.
        for _i in range(self.params.preenc_encoder_steps):
            loss, counter = 0.0, 0
            for r, (state_batch, observation_batch) in enumerate(self.replay_buffer):
                _preenc_opt.zero_grad()
                _x = self.learner_pol._do_flat(observation_batch.type(torch.double).to(self.params.device))
                _y = self.learner_pol.encoder(_x)
                _l = _preenc_loss(_y, state_batch.type(torch.double).to(self.params.device))
                _l.backward()
                _preenc_opt.step()
                counter += 1
                loss += _l.detach()
            if _i % 5 == 0: print('Step {: 02d}: Encoder Loss: {: >0.5f}.'.format(_i, loss.item() / counter))

        # Set the flag that we are pretrained.
        self.learner_pol.encoder.PRETRAINED = True

        # Send everything back to the CPU and save.
        self.learner_pol.encoder.to('cpu')
        # try:
        #     torch.save(self.learner_pol.encoder.state_dict(), "tests/worm_tests/results/AdaptAsymDagger/" + self.params.env_name + "/ete_encoder.pt")
        # except:
        #     pass
        torch.save(self.learner_pol.encoder.state_dict(), self.params.log_dir + '/ete_encoder.pt')


    def _test_encoder(self):
        """
        Short script just to test the performance of the encoder.  It will use/generate
        a batch of data and test the accuracy, precision and recall of each output.

        For partial observations, the performance of the encoder will be lacklustre,
        but for full_observe the encoder should return A/P/R of one.  This shows the
        encoder is capable of encoding the state.  Then running RL will converge in a
        similar time to RL(MDP) showing that everything is working.
        :return:
        """

        try:
            # Test the encoder.
            if not self.params.preenc_train_new_encoder:
                _, _preenc_batch, _ = sample_batch(self.expert_sampling_dist, self.env, self.params, eval_=True)
                self.replay_buffer.push(deepcopy(_preenc_batch), )
            t, x = self.replay_buffer.sample(len(self.replay_buffer) - 1)
            _x = self.learner_pol._do_flat(x)
            y = self.learner_pol.encoder(_x.type(torch.double).to(self.params.device))

            y = (y > 0.5).type(torch.int)
            t = t.type(torch.int)

            total = t.numel()
            acc = float(torch.sum(y == t)) / total
            recall = float(torch.sum(y[t == 1] == t[t == 1])) / t[t == 1].numel()
            precision = float(torch.sum(y[y == 1] == t[y == 1])) / t[y == 1].numel()

            print('Encoder Acc: {}'.format(acc))
            print('Encoder Pre: {}'.format(precision))
            print('Encoder Rec: {}'.format(recall))
        except:
            print('Soft warning: encoder test code failed.')
            pass
