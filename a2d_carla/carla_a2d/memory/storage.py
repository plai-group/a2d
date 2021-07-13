import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import deque
import random
import time
from copy import deepcopy
import numpy as np

# main storage class
class RolloutStorage(object):
    """
    This class is used to store states, actions, masks, hidden states, rewards,
    values (computed by outer function), and associated returns. Primarily
    relevent for on-policy reinforcement learning algorithms for faster access
    and transformations. Other methods are used to store larger batches of
    examples, as in the case of online imitation learning with DAgger, as well
    as off-policy reinforcement learning algorithms like soft actor critic.
    There are some indexing ideosynchronies, and after each sampling step, the
    pointer to the current state must be forcably moved. This is to allow for
    the use of streaming data as is often used in algorithms like PPO and A2C.

    Attributes
    ----------
    num_steps: int
    Number of environment steps per proccess to be stored at each iteration.
    This number should align with the number of environment steps used for each
    policy update divided by the number of procceses.

    num_processes : int
    Number of procceses which will be run in parallel for each environment update.
    This needs to align with the number of environments used in the vectorized
    pytorch environment wrapper from the baselines library.

    obs_space : gym.spaces.Dict
    Observation space as defined in openai gym. In our case we assume that this
    obersvation space is a dictionary so that we can mix and match different
    types of observations, and include different types of state information
    without modification of the codebase. More information can be found at:
    https://github.com/openai/gym/blob/master/gym/spaces/dict.py

    action_space:  gym.spaces.Box
    The action space again draws from the format of openai gym and in the case
    of carla is defined by a contiuas vector valued quantity. All environments
    assume use this format so here it is just stored as a vector.

    recurrent_hidden_state_size: gym.spaces.Dict
    (Under Construction....)

    Methods
    -------
    to(device)
    Moves all data in replay buffer to a specific device.

    insert(next_obs, next_recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, next_masks, next_bad_masks, k)
    Adds the next set of information (e.g. state,action,reward ect.) to the
    current vectorized buffer of information.

    after_update()
    This function takes the final state of allproccesses and moves it to the
    initial state index within the replay buffer. This allows for streaming
    state information for A2C, PPO, ect.

    compute_returns(next_value, use_gae, gamma, gae_lambda, use_proper_time_limits)
    This algorithm takes the current batch, and computes the returns associated
    with each state for use in advantage estimation. Currently, this setup allows
    for computation of normal returns, as well as GAE returns, and truncated time
    horizon returns (e.g. we do not bootstrap with the value function at the end)
    """
    def __init__(self, num_steps, num_processes, obs_space, action_space,
                 recurrent_hidden_state_size):
        if obs_space.shape is None:
            spaces_names = list(obs_space.spaces.keys())
            spaces_shapes = [obs_space.spaces[key].shape for key in spaces_names]
            spaces_dict = {key:value for (key, value) in zip(spaces_names,spaces_shapes)}
        else:
            spaces_names = ['obs']
            spaces_shapes = [obs_space.shape]
            spaces_dict = {key:value for (key, value) in zip(spaces_names,spaces_shapes)}
        self.spaces_dict = spaces_dict
        self.num_processes = num_processes
        self.num_steps = num_steps
        self.obs_keys = spaces_names
        self.obs = {key:torch.zeros(num_steps + 1, num_processes, *obs_shape) for (key, obs_shape) in zip(spaces_names,spaces_shapes)}
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for key in self.obs_keys:
            self.obs[key] = self.obs[key].to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, next_obs, next_recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, next_masks, next_bad_masks, k=None):
        for key in self.obs_keys:
            self.obs[key][self.step + 1].copy_(next_obs[key])
        self.recurrent_hidden_states[self.step + 1].copy_(next_recurrent_hidden_states)
        self.actions[self.step].copy_(actions.view(self.actions[self.step].size()))
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(next_masks)
        self.bad_masks[self.step + 1].copy_(next_bad_masks)
        if k is not None:
            assert self.step == k
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        for key in self.obs_keys:
            self.obs[key][0].copy_(self.obs[key][-1])
        # self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self, next_value, use_gae,
                        gamma, gae_lambda, use_proper_time_limits=True):

        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step +1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

# replay buffer
class TransitionReplayMemory:
    """
    This class is used to store all trajectory information in a larger buffer
    for use in off-policy learning algorithms.

    Attributes
    ----------
    size: int
    This provides the maximum allowable size of the first in first out replay
    buffer. Becuase this data will not be all kept on GPU this can be quite
    large, though it is currently not well optimized.

    Methods
    -------
    push(rollouts)
    Adds the next set of information (e.g. state,action,reward ect.) to the
    current buffer of information.

    sample()
    This function takes the final state of allproccesses and moves it to the
    initial state index within the replay buffer. This allows for streaming
    state information for A2C, PPO, ect.

    get_len(batch_size, device)
    Samples uniformly random subset from the data replay buffer and pushed it
    to the desired device before returning it.
    """
    def __init__(self, size):
        self.size = size
        self.memory = deque([],maxlen=size)
        self.spaces_names = None
        self.spaces_shapes = None
        self.spaces_dict = None

    def push(self, rollouts):
        # obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, self.num_processes, _ = rollouts.rewards.size()
        # reshaped_obs = rollouts.obs[:-1].view(-1, *obs_shape)
        self.spaces_names = list(rollouts.obs.keys())
        self.spaces_shapes = [rollouts.obs[key].size()[2:] for key in self.spaces_names]
        self.spaces_dict = {key:value for (key, value) in zip(self.spaces_names,self.spaces_shapes)}
        reshaped_obs = {key:rollouts.obs[key][:-1].view(-1, *self.spaces_dict[key]) for key in self.spaces_names}
        next_reshaped_obs = {key:rollouts.obs[key][1:].view(-1, *self.spaces_dict[key]) for key in self.spaces_names}
        # reshape examples into list of states
        obs_list = [{key:reshaped_obs[key][i,...].detach() for key in reshaped_obs.keys()} for i in range(num_steps*self.num_processes)]
        next_obs_list = [{key:next_reshaped_obs[key][i,...].detach() for key in next_reshaped_obs.keys()} for i in range(num_steps*self.num_processes)]
        rewards = rollouts.rewards.view(-1).detach()
        actions = rollouts.actions.view(-1, action_shape).detach()
        masks = rollouts.masks[:-1, ...].view(-1).detach()
        # put em togather
        new_info = [v for v in zip(obs_list, next_obs_list, rewards, actions, masks)]

        # extend the mem
        if len(self.memory) == 0:
            self.memory = deque(new_info,maxlen=self.size)
        else:
            self.memory.extendleft(new_info)

    def sample(self, batch_size, device='cpu'):
        # stop over-sampling
        if batch_size == np.Inf:
            sample_size = int(len(self.memory))
        else:
            sample_size = min(int(batch_size), int(len(self.memory)))
        assert sample_size > 0
        batch = random.sample(self.memory, sample_size)
        # return reshaped obs
        reshaped_obs = {}
        next_reshaped_obs = {}
        for key in self.spaces_names:
            reshaped_obs[key] = torch.stack([batch_[0][key] for batch_ in batch],dim=0).to(device)
            next_reshaped_obs[key] = torch.stack([batch_[1][key] for batch_ in batch],dim=0).to(device)
        rewards = torch.stack([batch_[2] for batch_ in batch],dim=0).to(device)
        actions = torch.stack([batch_[3] for batch_ in batch],dim=0).to(device)
        masks = torch.stack([batch_[4] for batch_ in batch],dim=0).to(device)
        # restack into single dict
        return reshaped_obs, next_reshaped_obs, rewards, actions, masks

    def get_len(self):
        return len(self.memory)

# replay buffer
class StateReplayMemory:
    """
    This class is used to store all trajectory information in a larger buffer
    for use in online imiitation learning algorithms like DAgger.

    Attributes
    ----------
    size: int
    This provides the maximum allowable size of the first in first out replay
    buffer. Becuase this data will not be all kept on GPU this can be quite
    large, though it is currently not well optimized.

    Methods
    -------
    push(rollouts)
    Adds the next set of information (e.g. state,action,reward ect.) to the
    current buffer of information.

    sample()
    This function takes the final state of allproccesses and moves it to the
    initial state index within the replay buffer. This allows for streaming
    state information for A2C, PPO, ect.

    get_len(batch_size, full_buffer)
    Samples uniformly random subset from the data replay buffer and pushed it
    to the the pre-defined device. full_buffer indicates that the entire,
    vectorized buffer should be returned to be parsed.
    """
    def __init__(self,size):
        self.size = size
        self.memory = deque([],maxlen=size)
        self.spaces_names = None
        self.spaces_shapes = None
        self.spaces_dict = None

    def push(self, rollouts):
        # obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        # reshaped_obs = rollouts.obs[:-1].view(-1, *obs_shape)
        self.spaces_names = list(rollouts.obs.keys())
        self.spaces_shapes = [rollouts.obs[key].size()[2:] for key in self.spaces_names]
        self.spaces_dict = {key:value for (key, value) in zip(self.spaces_names,self.spaces_shapes)}
        reshaped_obs = {key:rollouts.obs[key][:-1].view(-1, *self.spaces_dict[key]) for key in self.spaces_names}
        # reshape examples into list of states
        obs_list = [{key:reshaped_obs[key][i,...].detach() for key in reshaped_obs.keys()} for i in range(num_steps*num_processes)]
        # extend the mem
        self.memory.extendleft(obs_list)

    def sample(self, batch_size, full_buffer=False):
        if full_buffer:
            batch = deepcopy(self.memory)
            random.shuffle(batch)
            sample_size = self.get_len()
        else:
            batch = random.sample(self.memory, sample_size)
            sample_size = min(batch_size,self.get_len())
        # return reshaped obs
        reshaped_obs = {}
        for key in self.spaces_names:
            reshaped_obs[key] = torch.stack([batch[i][key] for i in range(sample_size)],dim=0)
        # restack into single dict
        return reshaped_obs

    def get_len(self):
        return len(self.memory)

#
class DictDataset(torch.utils.data.TensorDataset):
    """
    This class was created to take advantage of built in pytorch multi-proc
    for minibatch sgd on dictionary datasets.

    Attributes
    ----------
    data_dict: dict
    Dictionary of tensor data, where each index-0 indicates the example within
    the set of data.

    Methods
    -------
    __len__()
    returns the number of examples.

    __getitem__(batch_size, full_buffer)
    Grabs item at specified index (in this case a set of key values at an index).
    """
    def __init__(self, data_dict):
        self.data = data_dict
        self.data_keys = data_dict.keys()
    def __getitem__(self, index):
        return {key:self.data[key][index] for key in self.data_keys}
    def __len__(self):
        return [len(self.data[key]) for key in self.data_keys][0]
