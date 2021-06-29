# This code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Licence.  The full licence and information is available at:
# https://github.com/plai-group/a2d/blob/master/LICENCE.md
# © Andrew Warrington, J. Wilder Lavington, Adam Scibior, Mark Schmidt & Frank Wood.
# Accompanies the paper: Robust Asymmetric Learning in POMDPs, ICML 2021.

import timeit
import time
import gym
import torch
import shutil
import numpy as np

import matplotlib.pyplot as plt

from copy import deepcopy

from a2d.util.sampler import sample_batch
from a2d.util.inits import init_logger
from a2d.util.rl_update import trpo_step
from a2d.A2D_base_class import A2D_base


class AdpAsymDagger(A2D_base):
    """
    ADAPTIVE ASYMMETRIC DAGGER IMITATION LEARNING CLASS:
    Takes in a environment along with args defined by get_args, to train
    an expert (if none exist). Then uses this expert to train an
    image based model that is used by calling render on the enviroment.

    Note that I have separated out this class into two classes:
    AdpAsymDagger (This class): contains all of the ``interesting'' functionality.
    A2D_base: contains all the boring functionality, mainly logging and printing.

    I have a really hard time reading / using long python files.

    All of the initialization stuff is done in the base class.  This class then
    just adorns the base class with the additional functionality.
    """

    def __init__(self, args, img_stack=1, img_size=42, wrappers=None):
        """
        This just calls the base class initializer.
        :param args:
        :param img_stack:
        :param img_size:
        :param wrappers:
        """
        super().__init__(args, img_stack=img_stack, img_size=img_size, wrappers=wrappers)


    def compute_loss(self, _states, _observations):
        """
        Compute the loss of the minibatch that will be minimised by DAgger.
        :param _states:             (Tensor):   States to evaluate expert at.
        :param _observations:       (Tensor):   Observations to evaluate learner at.
        :return:                    (FloatTensor):  Loss of minibatch (with grad info).
        """
        # Get the log-prob distribution over actions from expert and agent.
        learner_lp = self.learner_pol(_observations)
        expert_lp = self.expert_pol(_states)

        # Compute loss.
        loss = self.discrete_kl_loss(learner_lp, expert_lp.detach())
        return loss


    def mini_batch_update(self, update=True, augmentate=None):
        """
        Perform a single epoch of mini-batch update. Now uses inbuilt replay buffer.
        :param update:          (bool):         apply the update.  (False can be used to measure divergence).
        :param augmentate:      (bool/None):    augmentate data.  =False means no augmentation (for evaluation).
        :return:                (float):        average loss across the epoch.
        """
        # compute avg loss
        average_loss, counter = 0., 0.

        # Override whether we are augmentating.
        if augmentate is not None:
            self.replay_buffer._do_augmentate = augmentate

        # Update policy with mini-batch steps.
        for r, (state_batch, observation_batch) in enumerate(self.replay_buffer):

            # compute loss
            loss = self.compute_loss(state_batch.double().to(self.params.device),
                                     observation_batch.double().to(self.params.device))

            # if we want to update the model
            if update:
                for _k in self.optims.keys():
                    if self.optims[_k] is not None:
                        self.optims[_k].zero_grad()     # zero grad
                loss.backward()                         # back-prop
                self.optims['learner_pol'].step()       # step

            loss = loss.detach().to('cpu')              # now detach
            average_loss += loss                        # update loss average
            counter += 1                                # set iter

        if augmentate is not None:
            self.replay_buffer._do_augmentate = None
        avg_loss = average_loss / counter
        return avg_loss


    def projection_step(self, verbose=False):
        """
        Project learner onto expert with mini-batch sgd.
        :param verbose:     (bool):     more textual output.
        :return:
        """
        _st = timeit.default_timer()
        self.to_device()
        losses, avg_loss = [], 0.0

        # Loop over epochs.
        for i in range(self.max_proj_iter):
            avg_loss = self.mini_batch_update(verbose)
            losses.append(avg_loss)

        # Give some output.
        if verbose:
            print('Internal projection losses: ' + ' '.join(['{: >0.4f}'.format(_s) for _s in losses]))

        # Inscribe stuff and send back to CPU.
        self.current_projection_loss = avg_loss
        self.timings['proj'] += timeit.default_timer() - _st
        self.to_cpu()
        return avg_loss


    def rl_step(self, batch, networks_to_update, verbose=False):
        """
        Take an RL step in the prescribed networks.
        :param batch:       (Memory):       trajectories gathered from the rollout.
        :param verbose:     (bool):         more text output?
        :return:
        """

        self.to_device()

        # Apply rl step to expert.
        if 'expert' == networks_to_update:
            trpo_step(self, batch, 'expert')

        # Apply rl step to the learner value function if need be.
        if 'learner_val' == networks_to_update:
            trpo_step(self, batch, 'learner_val')

        # Otherwise apply to the whole learner.
        elif 'learner' == networks_to_update:
            trpo_step(self, batch, 'learner')

        self.to_cpu()
        return None


    def get_data(self, sampling_dist='mixture', store=True, verbose=False, eval_=False):
        """
        Rollout under the specified action distribution.
        :param sampling_dist:   (string):       policy to rollout under ('expert','learner','mixture').
        :param store:           (bool):         add samples to the persistent replay buffer?
        :param verbose:         (bool):         more text output.
        :param eval_:           (bool):         eval_=True means deterministic rollouts using the argmax of
                                                policy.  =False means stocastic rollouts using a sample.
        :return:
        """
        self.to_device()
        _st = timeit.default_timer()

        # sample from env then add to buffer
        with torch.no_grad():

            # Get the correct sampling distribution.
            if sampling_dist == 'mixture':
                dist = self.mixed_sampling_dist
            elif sampling_dist == 'learner':
                dist = self.learner_sampling_dist
            elif sampling_dist == 'expert':
                dist = self.expert_sampling_dist
            else:
                raise Exception('define appropriate sampling distribution.')

            reward_batch, batch, expert_action = sample_batch(dist, self.env, self.params, eval_=eval_)

        # Add to buffer.
        if store:
            self.replay_buffer.push(deepcopy(batch), deepcopy(expert_action))
        else:
            self.replay_buffer.examples_seen += len(batch.state)  # At least increment the number of examples seen.
        if verbose: print('average {} (eval_={}) reward: {: >06.2f}. : '.format(sampling_dist, eval_, reward_batch))

        self.to_cpu()
        self.timings['data'] += timeit.default_timer() - _st
        return batch


    def step(self, log_iter, rl_update=True, proj_update=True, verbose=True, sampling_dist='mixture',
             store_=True, networks_to_update=None):
        """
        Take a single step.  This will gather data by rolling out (self.get_data) and then take the
        appropriate steps.  The steps are specified by the networks_to_update variable.

        An RL update step to the expert will be done first.  Then an RL update to either the whole
        learner (if doing RL in learner) or just the value function of the learner.  Finally, a
        projection step may be taken projecting the learner onto the expert policy.

        :param log_iter:            (int):      iteration number.
        :param rl_update:           (bool):     apply RL update to networks_to_update.
        :param proj_update:         (bool):     apply projection update to learner.
        :param verbose:             (bool):     more output.
        :param sampling_dist:       (string):   distribution to sample data under.  {'expert', 'learner', 'mixture'}
        :param store_:              (bool):     Add the samples to the persistent replay buffer?
        :param networks_to_update:  List of strings (Default: None -> ['expert', 'learner_val'] \\in {'expert',
                                    'learner_val', 'learner'}) indicating which  networks to update in the RL step.
                                    'expert' means that the WHOLE EXPERT will be updated (value function and policy).
                                    'learner_val' (superceeds 'learner') will only update the value function.
        :return:
        """
        
        # create data loader.
        batch = self.get_data(sampling_dist, store=store_)

        if rl_update:

            if networks_to_update is None:
                # These are the default arguments for A2D -- update the expert and the learner value function while
                # fixing the rest of the networks.
                networks_to_update = ['expert', 'learner_val']

            # Take an expert RL step.
            if 'expert' in networks_to_update:
                # Update only the MDP (policy & value functions).
                _st = timeit.default_timer()
                self.rl_step(batch, networks_to_update='expert')
                self.timings['rl_ex'] += timeit.default_timer() - _st

            # Take a value function step in the Learner.
            if 'learner_val' in networks_to_update:
                _st = timeit.default_timer()
                self.rl_step(batch, networks_to_update='learner_val')
                self.timings['rl_le_val'] += timeit.default_timer() - _st

            # Take a step in the whole learner.
            elif 'learner' in networks_to_update:
                _st = timeit.default_timer()
                self.rl_step(batch, networks_to_update='learner')
                self.timings['rl_le'] += timeit.default_timer() - _st

        # Take a projection step in the learner policy (AIL/A2D).
        if proj_update:
            self.projection_step(verbose=verbose)

        # Log and print.
        self.logging(log_iter)


    def A2D(self, verbose=True, log_it=True):
        """
        take n asymmetric dagger steps with an on-line rl step
        :param verbose:
        :param log_it:
        :return:
        """

        step_args = {'sampling_dist': 'mixture',
                     'networks_to_update': ['expert', 'learner_val'],
                     'verbose': verbose}

        # Set beta initially.  If we are decaying beta immediately.
        if self.beta_decay == 0:
            self.beta = 0.0
        else:
            self.beta = 1.0

        for i_episode in range(1, self.params.iters):                           # training loop
            self.step(i_episode, **step_args)                                   # Step algorithm.
            self.beta *= self.beta_decay                                        # Update beta.


            # # NOTE - these steps investigate reducing \lambda during optimization.
            # # This reduces the bias in the MC estimator and can lead to more stable
            # # convergence.  This was _not_ included in the results presented in the
            # # main paper, but is briefly commented on in the supplementary materials.
            # # Investigating this further is a promising line of future research.
            # # Please reach out if you are interested in this.
            # if i_episode == 100:
            #     print("\nReducing lambda_ from {}".format(self.params.lambda_))
            #     self.params.lambda_ -= 0.1
            #     print("to {}\n".format(self.params.lambda_))
            #
            # if i_episode == 300:
            #     print("\nReducing lambda_ from {}".format(self.params.lambda_))
            #     self.params.lambda_ -= 0.1
            #     print("to {}\n".format(self.params.lambda_))


            # Force disconnect the expert at low \beta levels.
            if self.beta < 0.001:
                self.beta = 0.0


    def AD(self, verbose=True, log_it=True):
        """
        take n asymmetric dagger steps
        :param verbose:
        :param log_it:
        :return:
        """
        step_args = {'rl_update': False,
                     'sampling_dist': 'mixture',
                     'verbose': verbose}

        self.beta = 1.0                                                         # Fix beta initially.
        for i_episode in range(1, self.params.iters):                           # Training loop.
            self.step(i_episode, **step_args)                                   # Step algorithm.
            self.beta *= 0.0                                                    # (Aggressively) update beta.


    def RL(self, verbose=True, log_it=True):
        """
        straight RL on the {M,POM}DP
        :param verbose:
        :param log_it:
        :return:
        """
        if self.params.render_type == 'state':
            # Use expert.
            self.beta = 1.
            dist = 'expert'
            update = ['expert']

        else:
            # Use learner.
            self.beta = 0.
            dist = 'learner'
            update = ['learner']

        step_args = {'proj_update': False,
                     'sampling_dist': dist,
                     'store_': False,
                     'networks_to_update': update,
                     'verbose': verbose}

        for i_episode in range(1, self.params.iters):                           # Training loop.
            self.step(i_episode, **step_args)                                   # Learning in Learner directly.


    def ARL(self, verbose=True, log_it=True):
        """
        straight RL on the POMD with asymmetric value function.
        :param verbose:
        :param log_it:
        :return:
        """

        step_args = {'proj_update': False,
                     'sampling_dist': 'learner',
                     'store_': False,
                     'networks_to_update': ['learner'],
                     'verbose': verbose}

        self.beta = 0.0                                                         # Ignore expert.
        self.beta_decay = 0.0                                                   # Always ignore expert.
        for i_episode in range(1, self.params.iters):                           # Training loop.
            self.step(i_episode, **step_args)                                   # Step algorithm.


    def PretrainedEncoder(self, verbose=True):
        """
        Use a pretrained image encoder.  If the params.preenc_train_new_encoder=True then
        a new encoder will be trained using rollouts from under the current MDP policy.

        Else, the default encoder in the log file will be used.  Note that this can cause
        really unreliable performance as the precise conditions under which that encoder
        was trained are not recorded and hence may introduce "silent" errors.  The most
        reliable way to get around this is to train an MDP(RL) from scratch, and then
        train a new encoder.  On any machine with a GPU, training the encoder only takes
        on the order of minutes.
        :param verbose:
        :return:
        """

        # Set up the logger.
        self.logger = init_logger(self.params)

        # Either learn a new encoder, or load a pre-existing one.
        if self.params.preenc_train_new_encoder:
            self._train_encoder()
        else:
            # Load an encoder from the file.
            # NOTE - this behaviour is not especially "safe" as the conditions under which
            # this encoder were learned are not controlled.
            encoder_location = "tests/results/AdaptAsymDagger/" + self.params.env_name + "/ete_encoder.pt"
            print('>> Copy encoder from: {}'.format(encoder_location))

            # Copy the encoder in to the working directory.
            shutil.copy(encoder_location, self.params.log_dir + '/ete_encoder.pt')

            # Load in the encoder.
            self.learner_pol.encoder.load_state_dict(torch.load(encoder_location))

        # Test the encoder.
        if self.params.preenc_test_encoder: self._test_encoder()

        # Clean out the buffer.
        self.replay_buffer.hard_reset()

        # If we are doing Pre-Enc freeze the layers. otherwise, learn the encoder as well.
        # Otherwise it is just asymmetric RL.
        print('Doing Pre-Enc, freezing encoder')
        self.learner_pol.encoder.requires_grad_(False)

        # Print some stuff.
        print('Encoder Input:      {}'.format(self.learner_pol.encoder.obs_shape))
        print('Encoder Output:     {}'.format(self.learner_pol.encoder.feature_dim))

        # Force ignore the expert...
        self.beta = 0.0
        self.params.beta_decay = 0.0
        self.beta_decay = 0.0

        # NOTE - these are not used, they can be safely deleted for verifcation of that fact.
        # They can be retained because it allows one to test the divergence of expert and agent.
        delattr(self, 'expert_pol')
        delattr(self, 'expert_val')

        # Set up the args for the RL loop.
        step_args = {'proj_update': False,  # No projection step.
                     'sampling_dist': 'learner',  # Sample under the agent only.
                     'store_': True,  # Force store to allow divergence computation.
                     'networks_to_update': ['learner'],  # Update the learner (policy + vf).
                     'verbose': True, }  # Give us plenty of output for this.

        # Now enter in to standard loop.
        self.initial_log()
        for i_episode in range(1, self.params.iters):
            self.learner_pol.encoder.requires_grad_(False)  # Lets just make sure...
            self.step(i_episode, **step_args)

            # If we are being very verbose, output some parameters of the encoder.
            # This will confirm whether or not the encoder is being modified.
            if self.params.very_verbose:
                print('Encoder parameters: ' + str(next(self.learner_pol.encoder.parameters())).replace('\n', ''))

        self.final_log()    # Log the results.


    def dispatch_train(self):
        """
        Call the appropriate training subroutine for each training mode.
        :return:
        """
        if self.params.render_type == 'state':
            print('Training MDP agent using {}...'.format(self.params.mode))
        else:
            print('Training POMDP agent using {}...'.format(self.params.mode))

        verbose = self.params.verbose  # not self.params.cluster
        if self.params.mode == 'A2D':
            self.A2D(verbose=verbose)
        elif self.params.mode == 'AD':
            self.AD(verbose=verbose)
        elif self.params.mode == 'D':
            # Use the AD code but force any arguments here.
            self.params.render_type = 'state'
            self.AD(verbose=verbose)
        elif self.params.mode == 'RL':
            self.RL(verbose=verbose)
        elif self.params.mode == 'ARL':
            self.ARL(verbose=verbose)
        elif self.params.mode == 'AIL':
            self.beta_decay = 1.
            self.AD(verbose=verbose)
        else:
            raise Exception('Unrecognised mode.')


    def train(self, expert_pol, expert_val, learner_pol, learner_val, pre_train=True, verbose=True):
        """
        Main training function.
        :param expert_pol:
        :param expert_val:
        :param actor_critic_net:
        :param pre_train:
        :param verbose:
        :param verbose:
        :return:
        """

        # Setup logging.
        self.logger = init_logger(self.params)

        # Grab the networks and optimizers to use.
        self.define_networks(expert_pol, expert_val, learner_pol, learner_val)

        # If we are just doing straight RL, then we can delete some of the networks for clarity
        # This will also prevent some of the subroutines running and save a bit of compute.
        if self.params.mode == 'RL':
            if self.params.render_type == 'state':
                delattr(self, 'learner_pol')
                delattr(self, 'learner_val')
            else:
                delattr(self, 'expert_pol')
                delattr(self, 'expert_val')

        self.initial_log()          # Print the training setup.
        self.pretrain(pre_train)    # Do a pretraining step.
        self.dispatch_train()       # Call main training loop.
        self.final_log()            # Log the results.

