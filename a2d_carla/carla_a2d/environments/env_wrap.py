


# general
from copy import deepcopy
import string
import random
from collections import deque
import time
import math

# sci-py
import numpy as np
import scipy.misc
from PIL import Image
import torch
import torchvision
from torchvision.utils import save_image
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box as Continuous
import gym

# helpers
# from carla_a2d.utils.torch_utils import *
from carla_a2d.environments.env_utils import ZFilter, get_random_string, Identity
from carla_a2d.environments.env_utils import command_map, enforce_action_space, dict_mean

#
class CarlaEnvWrapper(gym.Wrapper):
    """
    Environment wrapper used to ensure that the correct state information is output
    to the sampling setup and policy learning algorithms for the codebase. This takes
    the general structure from the carla gym wrapper and ensure that everything we
    need (like fixed expert actions from the PID) are included in the state dictionary.
    This class also adds frame-stacking functionality and compact state and reward
    normalization. This wrapper currently functions with the following environments:
    1. plai-carla/OccludedPedestrian-v0
    2. plai-carla/OvertakingTruck-v0
    3. plai-carla/Corl2017-v0

    Each of these environments uses different reward surfaces, depending on the type
    of environment and what is available. Each environment class also requires
    a variety of different stopping conditions which are unique to that environment.
    The different types of rewards which can be included in all environments are:
    1. pid_reward
    2. waypoint_reward
    3. survive_bonus
    4. completed_reward

    Attributes
    ----------
    env: CarlaEnv
        Openai gym environment created by inverted ai, which natively runs the
        four environment/baselines.

    params : Namespace
        ArgParse Object which contains the set of parameters specifciied on
        initialization. These include general RL parameters, as well as optimization
        parameters, and logging info.

    env_idx : str
        Something to differentiate between environments when usinging multiproccessing.
        This string is usually just a random set of strings and numbers.

    Methods
    -------
    augment_reward(obs, reward, done, info, action, completed)
    This takes in the set of conditions specified a user, and adds them to the
    the reward each individually. All rewards added are between 0, and 5.

    check_end_conditions(is_done_, info)
    This function labels the type of end that occurred for logging and or
    debugging purposes. This is does not affect the reward conditions.

    modify_end_info(info)
    This takes the final information including end condition, cumulative reward,
    waypoint percentage, and time horizon and adds them to the info included
    when step is returned.

    reset()
    resets the environment by setting the environment state based on the initial
    distribution of the desired set of scenarios. This function only returns the
    state after reset, and does not return information, reward, or done flag.

    step(action)
    Update the state of the environment and return the reward, new state, whether
    the episode has terminated, and other information which we include for
    logging purposes.

    close()
    Closes environment properly by sending a signal to the lower level environment
    class which we are wrapping.

    """
    def __init__(self, env, params, env_idx):
        self.env = env
        self.params = params
        self.env_idx = env_idx
        # bird_view_dims
        self.birdview_x = params.birdview_res.width
        self.birdview_y = params.birdview_res.height
        self.birdview_z = params.expert_frame_stack
        # get frontview stuff
        self.frontview_x = params.frontview_res.width
        self.frontview_y = params.frontview_res.height
        self.frontview_z = params.agent_frame_stack
        # frame stacking
        self.info_stack =  params.aux_info_stack
        self.birdview_stack = params.expert_frame_stack
        self.frame_stack = params.agent_frame_stack
        # enviro filtering
        norm_states = params.norm_states
        norm_rewards = params.norm_rewards
        clip_obs = params.clip_obs
        clip_rew = params.clip_rew
        clip_obs = None if clip_obs < 0 else clip_obs
        clip_rew = None if clip_rew < 0 else clip_rew
        # general env_info
        if '_max_episode_steps' in self.env.__dict__.keys():
            self._max_episode_steps = self.env.get_attribute('_max_episode_steps')
        else:
            self._max_episode_steps = 2500
        self.action_space = self.env.action_space
        self.env.observation_space = self.env.observation_space
        self.max_steps = params.max_time_horizon
        # Environment type
        self.is_discrete = type(self.env.action_space) == Discrete
        # Number of actions
        action_shape = self.env.action_space.shape
        # set action shape
        self.num_actions = self.env.action_space.n if self.is_discrete else 0 \
                            if len(action_shape) == 0 else action_shape[0]
        self.counter = 0
        # something to indicate this en is used.
        self.key_tracker = get_random_string()

        # get max speed info
        try:
            speed_info = self.env.observation_space['speed']
        except:
            print('No speed included in state')
            speed_info = gym.spaces.Box(low=0., high=100., shape=(1,), dtype=float)

        # get compact state info
        try:
            self.compact_bounds = [torch.tensor(env.observation_space['compact_vector'].low),
                          torch.tensor(env.observation_space['compact_vector'].high)]
        except:
            print('No compact state in this set of scenarios')
            self.compact_bounds = [torch.tensor(-1.),torch.tensor(1.)]

        # different contrl envs.
        if params.env_name == 'plai-carla/OccludedPedestrian-v0':
            # x, y, yaw, speed, y coordinate of the truck, y coordinate of the oncoming vehicle + child dist
            stacked_compact = gym.spaces.Box(low=-1., high=1., shape=(self.birdview_stack,7,), dtype=float)
            stacked_ut_compact = gym.spaces.Box(low=-1., high=1., shape=(self.birdview_stack,7,), dtype=float)
        elif params.env_name == 'plai-carla/OvertakingTruck-v0':
            # x, y, yaw, speed, y coordinate of the truck, y coordinate of the oncoming vehicle
            stacked_compact = gym.spaces.Box(low=-1., high=1., shape=(self.birdview_stack,6,), dtype=float)
            stacked_ut_compact = gym.spaces.Box(low=-1., high=1., shape=(self.birdview_stack,6,), dtype=float)
        elif params.env_name == 'plai-carla/Corl2017-v0':
            stacked_compact = gym.spaces.Box(low=-1., high=1., shape=(self.birdview_stack,11,), dtype=float)
            stacked_ut_compact = gym.spaces.Box(low=-1., high=1., shape=(self.birdview_stack,11,), dtype=float)
        else:
            raise Exception('provide valid env_name: not '+params.env_name)

        #
        self.compact_shape = stacked_compact.shape

        # main inputs
        stacked_frames = gym.spaces.Box(low=0., high=1., shape=(self.frontview_z, self.frontview_y, self.frontview_x), dtype=float)
        stacked_birdview = gym.spaces.Box(low=0., high=1., shape=(self.birdview_z, self.birdview_x, self.birdview_y), dtype=float)
        stacked_commands = gym.spaces.MultiDiscrete(tuple([4 for _ in range(self.info_stack)]))
        stacked_speeds = gym.spaces.Box(low=speed_info.low[0], high=speed_info.high[0], shape=(self.info_stack,), dtype=float)

        # some aux losses info to predict
        times = gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=float)
        stacked_prev_actions = gym.spaces.Box(low=-1., high=1., shape=(self.info_stack,2), dtype=float)
        fixed_expert_actions = gym.spaces.Box(low=-1., high=1., shape=(2,), dtype=float)
        baseline_actions = gym.spaces.Box(low=-1., high=1., shape=(2,), dtype=float)
        crash_indicator = gym.spaces.Box(low=0., high=1., shape=(1,), dtype=float)
        invasion_indicator = gym.spaces.Box(low=0., high=1., shape=(1,), dtype=float)
        end_indicator = gym.spaces.Box(low=0., high=1., shape=(1,), dtype=float)

        position = gym.spaces.Box(low=0., high=1., shape=(3,), dtype=float)
        init_state_indicator = gym.spaces.Box(low=0., high=1., shape=(1,), dtype=float)

        # grey scale converter
        self.convert_grayscale = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                    torchvision.transforms.Grayscale(num_output_channels=1), torchvision.transforms.ToTensor()])

        # set the dictionary observation space
        self.observation_space = gym.spaces.Dict({'commands': stacked_commands,
                    'speeds': stacked_speeds, 'frames': stacked_frames,
                    'time': times, 'birdview': stacked_birdview,
                    'prev_actions': stacked_prev_actions, 'fixed_expert_actions': fixed_expert_actions,
                    'compact_vector': stacked_compact, 'ut_compact_vector': stacked_ut_compact,
                    'baseline_actions': baseline_actions, 'is_done': end_indicator})

        # Support for rewards normalization
        self.reward_filter = Identity()
        if self.params.norm_rewards:
            self.reward_filter = ZFilter(self.reward_filter, shape=(), center=False, clip=clip_rew)

        # Support for state normalization
        self.compact_state_filter = Identity()
        if self.params.norm_states:
            if self.params.compact_state_filter_key < 0:
                # assert 1==0
                self.compact_state_filter = ZFilter(self.compact_state_filter, shape=(self.compact_shape[-1]), center=True, clip=None)
            else:
                print('is this wrong?', self.params.save_dir)
                self.compact_state_filter = torch.load(self.params.save_dir+'/state_filters.pt')[self.params.compact_state_filter_key]
                self.compact_state_filter.reset()

        # something to log why the agent is dying
        self.reset_type = deque(maxlen=25)
        self.reward_contributions = deque(maxlen=500)

    def augment_reward(self, obs, reward, done, info, action, completed):

        # print('ok lets rock.', info)
        info['reward_contribution'] = {}

        # add penalty for crazy actions
        if self.params.action_penalty:
            info['reward_contribution']['action_penalty'] = 0.025 * -1 * np.log(abs(action[0])+1) #(1.-max(abs(action[0]), 1.)) # formerly 0.25
            info['reward_contribution']['action_penalty'] += 0.025 * -1 * np.log(abs(action[1])+1) #(1.-max(abs(action[1]), 1.))

        # add penalty for a collisions
        if self.params.collision_penalty:
            if info['collision']:
                info['reward_contribution']['collision_penalty'] = -1.
            else:
                info['reward_contribution']['collision_penalty'] = 0.

        # add penalty for a collisions
        if self.params.invasion_penalty:
            if info['invasion']:
                info['reward_contribution']['invasion_penalty'] = -1.
            else:
                info['reward_contribution']['invasion_penalty'] = 0.

        # add boundary checker
        action[0] = enforce_action_space(action[0], high=1., low=-1.)
        action[1] = enforce_action_space(action[1], high=1., low=-1.)

        # pid based reward
        if self.params.expert_reward:
            # info['reward_contribution']['movement_bonus'] = 0.01 * (1 - abs(1. - action[1]) / 2 ) * (1.-abs(action[0]))**2
            info['reward_contribution']['expert_bonus'] = 1. * (1 - abs(info['expert_action'][1] - action[1]) / 2 ) \
                    * (1.-abs(info['expert_action'][0] - action[0]) / 2)**2
            assert info['reward_contribution']['expert_bonus'] == info['reward_contribution']['expert_bonus']

        # update waypoints found stopping condition
        if self.params.waypoint_reward:
            if info['waypoints_found'] - self.current_waypoints_found == 0:
                info['reward_contribution']['waypoints_bonus'] = 0.
            else:
                info['reward_contribution']['waypoints_bonus'] = 1.

        # waypoint stopping conditions stuff
        if info['waypoints_found'] - self.current_waypoints_found == 0:
            self.waypoint_reward_timeout += 1
        else:
            self.waypoint_reward_timeout = 0
        self.current_waypoints_found = info['waypoints_found']

        # completed bonus
        if self.params.completed_reward:
            if completed:
                info['reward_contribution']['completed_bonus'] = 10.
            else:
                info['reward_contribution']['completed_bonus'] = 0.

        # survive_bonus
        if self.params.survive_reward:
            if (not done) or (completed):
                info['reward_contribution']['survive_bonus'] = 1.
            else:
                info['reward_contribution']['survive_bonus'] = 0.

        # survive_bonus
        if self.params.action_diff_penalty:
            info['reward_contribution']['action_diff_penalty'] = 0.025 * -1. * abs(action[0]-self.prev_action[0])
            info['reward_contribution']['action_diff_penalty'] += 0.025 * -1. * abs(action[1]-self.prev_action[1])
            self.prev_action = action

        #
        if self.params.env_name == 'plai-carla/OvertakingTruck-v0':
            info['reward_contribution']['passing_reward'] = info['original_compact_vector'][1] - info['original_compact_vector'][4]
            info['reward_contribution']['passing_reward'] *= -1 / 15

        # now add in
        reward = sum(info['reward_contribution'].values())

        # check if the reward is a tensor
        if not torch.is_tensor(reward):
            reward = torch.tensor(reward)

        # something to check what killed agent
        if done:
            self.reset_type.append(info)

        # now append the reward contributions storage
        self.reward_contributions.append(info['reward_contribution'])

        # now return everything
        return obs, reward, done, info

    def check_end_conditions(self, is_done_, info):

        # now set the stopper
        is_done, end_type = is_done_, None
        is_done, end_type = (True, 'time_out') if self.counter >= self._max_episode_steps else (is_done, end_type)

        # different contrl envs.
        if self.params.env_name == 'plai-carla/OccludedPedestrian-v0':
            is_done, end_type = (True, 'invasion') if (info['invasion'] and  self.counter > 1) else (is_done, end_type)
            is_done, end_type = (True, 'distance_violation') if (info['next_waypoint_distance'] >= 12.) else (is_done, end_type)
            is_done, end_type = (True, 'wp_time_out') if (self.waypoint_reward_timeout >= 35) else (is_done, end_type)
            is_done, end_type = (True, 'collision') if (info['collision'] and  self.counter > 1)  else (is_done, end_type)
        elif self.params.env_name == 'plai-carla/OvertakingTruck-v0':
            is_done, end_type = (True, 'wp_time_out') if (self.waypoint_reward_timeout >= 7) else (is_done, end_type)
            is_done, end_type = (True, 'lost-car') if (info['lost-car']) else (is_done, end_type)
            is_done, end_type = (True, 'distance_violation') if (info['next_waypoint_distance'] >= 25.) else (is_done, end_type)
            is_done, end_type = (True, 'collision') if (info['collision'] and  self.counter > 1)  else (is_done, end_type)
        elif  self.params.env_name == 'plai-carla/Corl2017-v0':
            is_done, end_type = (True, 'distance_violation') if (info['next_waypoint_distance'] >= 12.) else (is_done, end_type)
            is_done, end_type = (True, 'wp_time_out') if (self.waypoint_reward_timeout >= 35) else (is_done, end_type)
            is_done, end_type = (True, 'collision') if (info['collision'] and  self.counter > 1)  else (is_done, end_type)
        if not is_done:
            end_type = 'incomplete'
        elif end_type is None:
            end_type = 'completed'
        elif (end_type is None) and (not is_done):
            raise Exception('not sure how we arrived here: -> '+end_type)

        # now get the completion flag directly
        if end_type == 'completed':
            # assert self.current_waypoints_found > 9
            completed = 1.
        else:
            completed = 0.

        return is_done, end_type, completed

    def modify_end_info(self, info):
        # create an a
        info['additional_logs'] = {}
        # add
        info['additional_logs']['scenario_solved'] = info['endtype_info'][1]
        info['additional_logs']['reward_contributions'] = info['avg_reward_contrib']
        info['additional_logs']['episode_waypoints_hit'] = info['waypoints_info'][0]
        info['additional_logs']['episode_waypoints_percent'] = info['waypoints_info'][0]/info['waypoints_info'][1]
        # different end types
        info['additional_logs']['distance_violation_end'] = 1 if info['endtype_info'][0] == 'distance_violation_end' else 0
        info['additional_logs']['wp_time_out_end'] = 1 if info['endtype_info'][0] == 'wp_time_out_end' else 0
        info['additional_logs']['collision_end'] = 1 if info['endtype_info'][0] == 'collision' else 0
        info['additional_logs']['invasion_end'] = 1 if info['endtype_info'][0] == 'invasion_end' else 0
        info['additional_logs']['time_out_end'] = 1 if info['endtype_info'][0] == 'time_out_end' else 0
        info['additional_logs']['incomplete'] = 1 if info['endtype_info'][0] == 'incomplete' else 0
        # require that it finished and maxed reward.
        info['additional_logs']['masked_r_mean'] = info['endtype_info'][1]*info['endtype_info'][1]
        # return it
        return info

    def reset(self):

        # reset distance
        self.distance_from_goal = None

        # reward stuff
        self.current_waypoints_found = 0
        self.waypoint_reward_timeout = 0
        self.old_waypoint_dist = None

        # reset everything
        self.reward_filter.reset()
        self.compact_state_filter.reset()

        # Reset the state, and the running total reward
        start_state = self.env.reset()
        # update time horizonse
        if self._max_episode_steps is None or self.max_steps < self._max_episode_steps:
            self._max_episode_steps = self.max_steps
        # set true reward
        self.total_true_reward = 0.0
        # set time step checker
        self.counter = 0

        # get info
        frame = self.convert_grayscale(start_state['front_image'][0,...])
        birdview = self.convert_grayscale(start_state['birdview_image'][0,...])


        # get aux info
        speed = torch.tensor([start_state['speed']]) / 35.
        command = start_state['command']
        time = 0.
        command = torch.tensor([command_map(command)])
        compact_vector = torch.tensor(self.compact_state_filter(np.array(start_state['compact_vector']))).float()
        ut_compact_vector = torch.tensor(np.array(start_state['compact_vector'])).float()

        # init stack
        self.commands = [command for i in range(self.info_stack)]
        self.speeds = [speed for i in range(self.info_stack)]
        self.frames = [frame for i in range(self.frame_stack)]
        self.birdviews = [birdview for i in range(self.birdview_stack)]
        self.prev_actions = [torch.tensor(np.array([0.,0.])) for i in range(self.info_stack)]
        self.compacts = [compact_vector for i in range(self.birdview_stack)]
        self.ut_compacts = [ut_compact_vector for i in range(self.birdview_stack)]

        # now stack what needs to be stacked
        stacked_commands = torch.cat(self.commands, dim=0).detach()#.numpy()
        stacked_speeds = torch.cat(self.speeds).detach()#.numpy()
        stacked_frames = torch.cat(self.frames).detach()#.numpy()
        stacked_birdview = torch.cat(self.birdviews).detach()#.numpy()
        stacked_actions = torch.stack(self.prev_actions, dim=0).detach()#.numpy()
        stacked_compact = torch.stack(self.compacts, dim=0).detach()#.numpy()
        stacked_ut_compact = torch.stack(self.ut_compacts, dim=0).detach()#.numpy()

        # set pid action
        self.expert_action = [0.,0.]
        self.prev_action = [0.,0.]

        # set as a dictionary
        return {'commands': stacked_commands, 'speeds': stacked_speeds,
                'frames': stacked_frames, 'time': torch.tensor([time]),
                'birdview': stacked_birdview, 'prev_actions': stacked_actions,
                'fixed_expert_actions': torch.tensor([0.,0.]).detach(),
                'baseline_actions': torch.tensor([0.,1.]).detach(),
                # 'crash_ind': torch.tensor([0]), 'invasion_ind': torch.tensor([0]),
                'compact_vector': stacked_compact, 'ut_compact_vector': stacked_ut_compact,
                'is_done': torch.tensor([0])}

    def step(self, action):

        # step
        action = action.tolist()
        state, reward, is_done, info = self.env.step(action)

        #
        info['original_compact_vector'] = state['compact_vector']

        # update time
        self.counter += 1

        # bad transition filter
        if is_done and self._max_episode_steps >= self.counter:
            info['bad_transition'] = True

        # check if we have lost the truck
        if self.params.env_name == 'plai-carla/OvertakingTruck-v0':
            # print('how far.', self.counter, (torch.tensor(state['compact_vector'])[1] - torch.tensor(state['compact_vector'])[4]))
            if (torch.tensor(state['compact_vector'])[1] - torch.tensor(state['compact_vector'])[4]) > 15.:
                info['lost-car'] = True
            else:
                info['lost-car'] = False

        # check end conditions
        is_done, end_type, completed = self.check_end_conditions(is_done, info)

        # get info
        next_frame = self.convert_grayscale(state['front_image'][0,...])
        next_birdview = self.convert_grayscale(state['birdview_image'][0,...])

        # get aux info
        next_speed = torch.tensor([state['speed']]) / 35.
        next_command = torch.tensor([command_map(state['command'])])
        next_time = self.counter / self._max_episode_steps

        # transform compact statements
        next_compact_vector = torch.tensor(self.compact_state_filter(np.array(state['compact_vector']))).float()
        next_ut_compact_vector = torch.tensor(np.array(state['compact_vector'])).float()

        # remove last on list
        self.frames.pop()
        self.commands.pop()
        self.speeds.pop()
        self.birdviews.pop()
        self.prev_actions.pop()
        self.compacts.pop()
        self.ut_compacts.pop()

        # add on the new one
        self.frames = [next_frame] + self.frames
        self.commands = [next_command] + self.commands
        self.speeds = [next_speed] + self.speeds
        self.birdviews = [next_birdview] + self.birdviews
        self.prev_actions = [torch.tensor(np.array(action))] + self.prev_actions
        self.compacts = [torch.tensor(np.array(next_compact_vector))] + self.compacts
        self.ut_compacts = [torch.tensor(np.array(next_ut_compact_vector))] + self.ut_compacts

        # now stack what needs to be stacked
        stacked_commands = torch.cat(self.commands, dim=0).detach()#.numpy()
        stacked_speeds = torch.cat(self.speeds, dim=0).detach()#.numpy()
        stacked_frames = torch.cat(self.frames, dim=0).detach()#.numpy()
        stacked_birdview = torch.cat(self.birdviews, dim=0).detach()#.numpy()
        stacked_actions = torch.stack(self.prev_actions, dim=0).detach()#.numpy()
        stacked_compact = torch.stack(self.compacts, dim=0).detach()#.numpy()
        stacked_ut_compact = torch.stack(self.ut_compacts, dim=0).detach()#.numpy()

        # set as a dictionary
        return_state = {'commands': stacked_commands, 'speeds': stacked_speeds,
                        'frames': stacked_frames, 'time': torch.tensor([next_time]),
                        'birdview': stacked_birdview, 'prev_actions': stacked_actions,
                        'fixed_expert_actions': torch.tensor(info['expert_action']).detach(),
                        'ut_compact_vector': stacked_ut_compact,
                        'baseline_actions': torch.tensor([0.,1.]).detach(),
                        'compact_vector':stacked_compact, 'is_done': torch.tensor([1*is_done])}

        self.expert_action = info['expert_action']
        # add current time to info
        info['time'] = self.counter

        # shape reward
        info['completed'] = completed
        return_state, reward, is_done, info = self.augment_reward(return_state, reward, is_done, info, action, completed)

        # log reward before its transformed
        self.total_true_reward += reward.item()

        # add run info to info block
        if is_done:
            # print('ahhhh', end_type, info['waypoints_found'], info['waypoints_total'])
            info['done'] = (self.total_true_reward, self.counter)
            # also add in way point info
            info['waypoints_info'] = (info['waypoints_found'], info['waypoints_total'], self._max_episode_steps)
            # how it ended
            info['endtype_info'] = (end_type, completed)
            # the avg reward contributions
            info['avg_reward_contrib'] = dict_mean(self.reward_contributions)
            # combine into additional info
            info = self.modify_end_info(info)

        # filter reward
        _reward = self.reward_filter(reward.item())

        # return
        return return_state, _reward, is_done, info

    def close(self):
        self.env.close()
