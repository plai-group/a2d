# This code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Licence.  The full licence and information is available at:
# https://github.com/plai-group/a2d/blob/master/LICENCE.md
# © Andrew Warrington, J. Wilder Lavington, Adam Scibior, Mark Schmidt & Frank Wood.
# Accompanies the paper: Robust Asymmetric Learning in POMDPs, ICML 2021.

# general imports
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from a2d.util.replay_buffer import Memory_POMDP
from a2d.environments.env_wrap import _my_render


def sample_batch(action_dist, env, args, eval_=False, store=True, full_trajectory=False, SAVE_RENDERS=False, ITER=None):
    """
    AW - sample some data / perform a rollout.
    :param action_dist:
    :param env:
    :param args:
    :param eval_:
    :param store:
    :param expert_policy:
    :param agent_policy:
    :return:
    """

    def _configure_im_save():
        SHOW = False
        if SHOW:
            plt.switch_backend('macosx')

        if type(ITER) is str:
            dir_to_save = args.log_dir + '/imgs_' + str(ITER)
        else:
            dir_to_save = args.log_dir + '/imgs_{0:03d}'.format(ITER)
        try: os.mkdir(dir_to_save)
        except: pass
        return dir_to_save, SHOW

    def _save_im():
        plt.close('all')

        # plt.switch_backend('macosx')

        for _k in ['full_observe', 'observe']:

            _im = _my_render(env, 'minigrid', tag='minigrid', obs_type=_k)

            # if args.render_type == 'state':
            #     _im = _my_render(env, 'minigrid', tag='minigrid', obs_type='full_observe')
            # else:
            #     _im = _my_render(env, 'minigrid', tag='minigrid', obs_type='observe')

            plt.figure(figsize=(4, 4))
            plt.imshow(_im.to('cpu').permute(1, 2, 0))
            _dist_string = "[" + ', '.join(['{:> 5.3f}'.format(_f.to('cpu').numpy()) for _f in dist.squeeze()]) + "]"
            _action_string = ['D', 'R', 'U', 'L'][action]
            # plt.title('Iter {}  |  Reset {}  |  Dist {}  |  Action {}/{}'.format(ITER, reward_sum == 0, _dist_string, action, _action_string))

            plt.axis('off')
            plt.tight_layout()

            plt.savefig(dir_to_save + '/img_{0}_{1:05d}.png'.format(_k, num_steps), dpi=200)

            if type(ITER) == str:
                _str = '/im0/img_0_' + ITER
            else:
                _str = '/im0/img_0_{0:05d}.png'.format(ITER)
            if num_steps == 0: plt.savefig(args.log_dir + _str, dpi=200)
            if SHOW: plt.pause(0.00001)


    # Slightly dirty code for dumping out some trajectory renderings.
    # Pictures are pretty, aren't they.
    if SAVE_RENDERS:
        dir_to_save, SHOW = _configure_im_save()


    # Initialise the memory and storage variables.
    memory = Memory_POMDP()
    num_steps, reward_batch, num_episodes, expert_actions = 0, 0, 0, []
    enough_samples = False


    # Not backpropping through this, so no_grad makes everything faster
    # and reduces the memory overheads.
    with torch.no_grad():

        # Main sampling loop.
        while num_steps < args.batch_size:

            # Reset for a new episode.
            env.reset()
            reward_sum = 0

            # iterate.
            for t in range(1, 5000):

                # Get the frame stack, if we are not forcing not rendering (used in MDP).
                _ = env.my_render()  # Call the render function.

                # But use the stacking in for clarity throughout.
                stored_observe = deepcopy(torch.stack(env.frames)).detach()
                if stored_observe.ndim == 4:
                    stored_observe = stored_observe.half().permute(0, 2, 3, 1)

                # Clone the state to make sure.
                state = deepcopy(_my_render(env, None, obs_type='state'))

                # Sample action.  If we are saving the renders, then we will also
                # extract some additional information.
                if SAVE_RENDERS:
                    action, expert_sample, dist = action_dist(state.to(args.device),
                                                              stored_observe.to(args.device),
                                                              eval_=eval_, return_dist=True)
                else:
                    action, expert_sample = action_dist(state.to(args.device),
                                                        stored_observe.to(args.device),
                                                        eval_=eval_)

                # Did we sample from the expert?
                expert_actions.append(expert_sample)

                # Process the action so that it is gym-amenable.
                action = action.data[0].detach().to('cpu').numpy()

                # Slightly dirty code for dumping out some trajectory renderings.
                # Pictures are pretty, aren't they.
                if SAVE_RENDERS:
                    _save_im()

                next_state, reward, done, _ = env.step(action)  # step environment
                reward_sum += reward  # collect reward
                mask = 0 if done else 1  # set done flag

                # Push the tuple to the lightweight memory.
                if store:
                    obs = deepcopy(stored_observe)
                    memory.push(np.asarray(state), action, mask, np.asarray(next_state), reward, obs)

                # If we have enough samples then break.
                num_steps += 1
                if (num_steps >= args.batch_size) and (not full_trajectory):
                    enough_samples = True
                    break

                # Check if the environment has reset.
                if done: break

            # Update some counters.
            num_episodes += 1
            reward_batch += reward_sum

            # If we have enough samples, then break out entirely.
            if enough_samples:
                break

        # Process the memory and compute some summary statistics.
        reward_batch /= num_episodes
        if store:
            batch = memory.sample()
            ea = torch.tensor(expert_actions).detach()
        else:
            batch = None
            ea = None

    # Stop any renderer that may be running.
    env.viewer = None
    env.close()
    return reward_batch, batch, ea
