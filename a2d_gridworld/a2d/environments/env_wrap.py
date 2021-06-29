# This code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Licence.  The full licence and information is available at:
# https://github.com/plai-group/a2d/blob/master/LICENCE.md
# © Andrew Warrington, J. Wilder Lavington, Adam Scibior, Mark Schmidt & Frank Wood.
# Accompanies the paper: Robust Asymmetric Learning in POMDPs, ICML 2021.

import gym
import torch
import matplotlib.pyplot as plt
import numpy as np

from gym.spaces.discrete import Discrete
from gym.spaces.box import Box as Continuous
from copy import deepcopy


def _my_render(env, env_name, tag='minigrid', resize=84, obs_type='observe'):
    """
    NOTE - DO NOT CALL THIS FUNCTION DIRECTLY FROM THE MAIN CODE.

    There is an accessor function EnvWrapper.my_render(..), EnvWrapper._call_my_render(..)
    that provides a sensible interface and also make use of the observation type inscribed
    into the class.  Calling this function directly may provide unstable results.

    The frame stack is then accessed via EnvWrapper.frames .

    :param env:
    :param env_name:
    :param tag:
    :param resize:
    :param obs_type:
    :return:
    """

    # All environments can return state via wrapper...
    if obs_type == 'state':

        # Grab the state from the env wrapper class.
        if torch.is_tensor(env.state):
            observation = env.state.detach().clone()
        else:
            observation = torch.tensor(env.state)
        return observation

    if tag == 'minigrid':

        # If we are returning an image, then call the MiniGrid render function but request
        # the render returns it as a matrix.
        if (obs_type == 'observe') or (obs_type == 'full_observe'):
            observation = env.env.render(mode='rgb_array', _key=obs_type)
            observation = torch.tensor(observation.copy()).T.detach()
        elif obs_type == 'partial_state':
            # Otherwise, just return it as a tensor.
            observation = torch.tensor(env.env.render(_key=obs_type)).detach()
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError('\nERROR: env_wrap.py->_my_render: '
                                  'Environment type not recognised.\n')

    return observation


class EnvWrapper(gym.Env):
    """
    Define a wrapper for a gym environment.
    This just allows us to add a bit of flexibility for tracking states _and_ observations.
    """

    def __init__(self, env, params, add_t_with_horizon=None, observation_type=None):
        """
        Initialize the wrapper.
        :param env:                 (GymEnv):           OpenAIGym environment to inscribe into wrapper.
        :param params:              (SimpleNamespace):  parameter object from A2Dagger_arguments.
        :param add_t_with_horizon:  Depreciated.
        :param observation_type:    Depreciated.
        """

        # Inscribe the environment and some of the parameters.
        self.env = env
        self.params = params
        self._max_episode_steps = self.env._max_episode_steps
        self.action_space = self.env.action_space
        self.DEFAULT_IM_SIZE = 84
        self.observation_space = self.env.observation_space

        self.render_type = params.render_type
        assert self.render_type in ['state', 'partial_state', 'observe', 'full_observe']

        # Default behaviour is to not flatten image observations.
        self.flatten = False

        # If it is a discrete env, force a single action.
        self.is_discrete = type(self.env.action_space) == Discrete
        assert self.is_discrete or type(self.env.action_space) == Continuous
        if self.is_discrete:
            self.env.action_space.shape = (self.env.action_space.n, )

        # Reset the state, and the running total reward
        self.state = torch.tensor(self.env.reset())

        # If the params specify a return type, then use that (comes from RL).
        if hasattr(params, 'return_type'):
            self.return_type = params.return_type
        else:
            self.return_type = None

        # Number of actions
        action_shape = self.env.action_space.shape
        assert len(action_shape) <= 1  # scalar or vector actions
        self.num_actions = self.env.action_space.n if self.is_discrete else 0 \
                            if len(action_shape) == 0 else action_shape[0]

        # Build frame stacking and observation stuff.
        self.frames = []
        self.frame_stack = params.frame_stack
        obs = self.my_render()
        if len(obs.shape) == 1:
            self.num_features = obs.shape[0]
        else:
            assert torch.max(obs) <= 255
            assert (obs.size()[0] == 3) or (obs.size()[0] == (3 * self.frame_stack))
            input_size = list(obs.size())
            self.num_features = input_size
            self.observation_space.shape = self.num_features
            self.observation_space.low = torch.zeros((self.num_features)).numpy()
            self.observation_space.high = 255*torch.ones((self.num_features)).numpy()
            self.observation_space.dtype = np.uint8(1.).dtype

        # Running total reward (set to 0.0 at resets).
        self.total_true_reward = 0.0


    def _gen_initial_frame_stack(self):
        """
        AW - wrap generation of the initial frame stack.
        :return:
        """
        if len(self.frames) == 0:
            obs = self._call_my_render()
            self.frames = [deepcopy(obs) for i in range(self.frame_stack)]


    def reset(self):
        # Reset the state, and the running total reward
        start_state = torch.tensor(self.env.reset())
        self.state = start_state  # Keep track of state, why not?
        self.total_true_reward = 0.0
        self.counter = 0.0

        # blitz any frame counter that exists.
        self.frames = []

        # Simplify return type by bundling into my_render().
        self._gen_initial_frame_stack()

        if self.return_type is not None:
            # Return type is flag used in WL code to return render for vanilla RL library.
            # manually overwrite the none flag to indicate that this behaviour should be
            # used.  Will break all other code...
            return_state = self.my_render()
        else:
            return_state = start_state

        return return_state


    def _call_my_render(self):
        if not hasattr(self.params, 'resize'):
            self.params.resize = self.DEFAULT_IM_SIZE
        return _my_render(env=self, env_name=self.params.env_name, tag=self.params.env_tag,
                          resize=self.params.resize, obs_type=self.params.render_type)


    def my_render(self, flatten=None):
        """
        AW - this function wraps the call to the static function my_render (defined above), and checks
        firstly that the buffer in this class is full before returning the stacked result.
        :return:
        """
        if flatten is None:
            flatten = self.flatten

        self._gen_initial_frame_stack()  # Check first that the frame buffer is full.
        next_obs = self._call_my_render()
        self.frames.pop()                                       # Pop the last frame.
        self.frames = [next_obs] + self.frames                  # add on the new one.
        stacked_obs = torch.cat(self.frames, dim=0).detach()    # get the stack.

        # reshape and convert to np.
        if flatten:
            return_state = stacked_obs.reshape(-1)
        else:
            return_state = stacked_obs

        return return_state


    def render(self, mode='human'):
        """
        AW - gym-compliant render function for calling the underlying render.
        :param mode:
        :return:
        """
        self.env.render(mode)


    def step(self, action):
        """
        AW - step the environment taking action.
        :param action:  (Action):   action to take.
        :return:        (Tuple):    iterated state/obs, instantaneous reward, end of episode, info dict.
        """

        # Step the environment.
        try:
            state, reward, is_done, info = self.env.step(action)
        except Exception as err:
            print(err)
            print('Error in iterating environment.')
            raise RuntimeError

        # AW - lets keep track of the state in the env as well, will
        # make switching the obs_type later on much more straightforward.
        self.state = state
        self.total_true_reward += reward
        self.counter += 1

        # If we are done, inscribe some info,
        if is_done:
            info['done'] = (self.total_true_reward, self.counter)

        if self.return_type is not None:
            # Return type is flag used in WL code to return render for vanilla RL library.  manually overwrite the
            # none flag to indicate that this behaviour should be used.  Will break all other code...
            return_state = self.my_render()
            info['state'] = state
        else:
            return_state = state

        # Make sure the return type is correct.
        if torch.is_tensor(return_state):
            return_state = return_state.detach().clone()
        else:
            return_state = torch.tensor(return_state)

        return return_state, float(reward), is_done, info
