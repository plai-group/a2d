
# old imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

# imports
import torch.autograd as autograd
from torch.autograd import Variable
import math
from torch.distributions import Normal
import numpy as np
import copy
from copy import deepcopy
import time
from torch import distributions as pyd

# library imports
from carla_a2d.models.distributions import DiagGaussian, FixedNormal
from carla_a2d.utils.torch_utils import init

class ActorCritic(nn.Module):
    """
    Primary actor critic class which handles all models which interactions with
    the environment. This houses both the differentiable expert and agent models
    which are used for training, and defines the mixture between value functions
    and policy used by A2D. Becuase of how A2D is defined, we are able to run
    different RL and AD algorithms as special cases of this class.

    Attributes
    ----------
    obs_space : gym.spaces.Dict
    Observation space as defined in openai gym. In our case we assume that this
    obersvation space is a dictionary so that we can mix and match different
    types of observations, and include different types of state information
    without modification of the codebase. More information can be found at:
    https://github.com/openai/gym/blob/master/gym/spaces/dict.py

    action_space: int
    Used to define the size of actions used by the policy. Here this value is
    used to directly define the distributoins over actions as parameterized
    distributions.

    base_kwargs: dict
    Set of arguments used to define the carla nueral network base. Usually this
    just contains the argumets parsed from the user input.

    Methods
    -------
    is_recurrent(x)
    Under Construction...

    recurrent_hidden_state_size()
    Under Construction...

    act(inputs, device, deterministic, force_expert_sample, reparam)
    Function to generate actions from the actor critic model through some mixture
    of expert and agent actions. Allows forced evaulation / sampling from the
    defined expert, as well as sampleing from a non-mixture distributoin using
    the reparameterization trick. Inputs just defines the current state.

    get_value(inputs, device)
    Returns value function, or mixture of value functions depending on actor
    critic hyper-parameters. Inputs just defines the current state.

    evaluate_actions(inputs, action, detach_encoder, device)
    Returns value function, or mixture of value functions depending on actor
    critic hyper-parameters. This also returns the log probabiluty under the model
    of the action, as well as the current distributional entropy for all states.
    Inputs just defines the current state.

    evaluate_expert_actions(inputs, action, detach_encoder, device)
    Returns expert value function This also returns the log probabiluty under the model
    of the action, as well as the current distributional entropy for all states
    under the expert policy.

    evaluate_agent_actions(inputs, action, detach_encoder, device)
    Returns agent value function This also returns the log probabiluty under the model
    of the action, as well as the current distributional entropy for all states
    under the agent policy.

    agent_expert_divergence(inputs)
    Computes the divergence between the expert distribution and trainee distribution.
    """
    def __init__(self, obs_shape, action_space, base_kwargs=None):
        super(ActorCritic, self).__init__()

        # some inits
        self.action_space = action_space
        self.observation_space = obs_shape
        if base_kwargs is None:
            base_kwargs = {}

        # nueral net base set
        base = CARLABase

        # nueral net base init
        if isinstance(obs_shape, gym.spaces.Dict):
            self.base = base(obs_shape, **base_kwargs)
        elif len(obs_shape) == 1:
            self.base = base(obs_shape[0], **base_kwargs)
        else:
            self.base = base(obs_shape, **base_kwargs)

        # policy distribution
        num_outputs = action_space.shape[0]
        self.agent_dist = DiagGaussian(self.base.agent_output_size, num_outputs, param_sd=self.base.command_dim)
        self.expert_dist = DiagGaussian(self.base.expert_output_size, num_outputs, param_sd=self.base.command_dim)
        self.fixed_expert = lambda fixed_actions, device: FixedNormal(fixed_actions.to(device), \
                torch.tensor(base_kwargs['params'].fixed_expert_sd).to(device) * torch.ones(fixed_actions.size()).to(device))

        # a2d stuff
        self.beta = torch.tensor(1.)
        self.use_learned_expert = None
        self.fixed_expert_sd = None
        self.sampling_dist = None
        self.critic_key = None

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def act(self, inputs, rnn_hxs=None, masks=None, device='cuda:1', deterministic=False, force_expert_sample=False, reparam=False):

        # get agent info
        if self.sampling_dist == 'agent':
            agent_value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
            agent_dist = self.agent_dist(actor_features)
        elif self.sampling_dist == 'expert':
            agent_value, expert_features, rnn_hxs = self.base.expert_forward(inputs, rnn_hxs, masks)
            agent_dist = self.expert_dist(expert_features)
        else:
            raise Exception('self.sampling_dist: '+self.sampling_dist)

        # get expert info
        if self.use_learned_expert:
            expert_value, expert_features, rnn_hxs = self.base.expert_forward(inputs, rnn_hxs, masks)
            expert_dist = self.expert_dist(expert_features)
        else:
            expert_value = None
            expert_dist = self.fixed_expert(inputs['fixed_expert_actions'], device)

        # get the action deterministically
        if deterministic:
            if not force_expert_sample:
                action = agent_dist.mode()
            else:
                action = expert_dist.mode()
        #
        elif self.beta >= torch.rand(1):
            if not reparam:
                action = inputs['fixed_expert_actions'] #expert_dist.sample()
            else:
                assert self.beta == 1.
                assert self.action_space.__class__.__name__ not in ["Discrete","MultiBinary"]
                action = expert_dist.rsample()
        else:
            if not reparam:
                action = agent_dist.sample()
            else:
                assert self.beta == 0.
                assert self.action_space.__class__.__name__ not in ["Discrete","MultiBinary"]
                action = agent_dist.rsample()

        # compute scaled beta for normalization
        if (self.beta > 0.) and (self.beta < 1.):
            agent_log_probs = agent_dist.log_probs(action) + torch.log(1-self.beta.to(device))
            expert_log_probs = (expert_dist.log_probs(action) + torch.log(self.beta.to(device)))
            log_probs = torch.cat([agent_log_probs, expert_log_probs],dim=1)
            action_log_probs = torch.logsumexp(log_probs,1).unsqueeze(1)
        elif self.beta == 0.:
            action_log_probs = agent_dist.log_probs(action)
        elif self.beta == 1. or force_expert_sample:
            action_log_probs = expert_dist.log_probs(action)
        else:
            raise Exception('something is guffed.')

        # now combine value functions
        if (expert_value is not None) and (self.critic_key=='mixture'):
            value = (1-self.beta.to(device))*agent_value + self.beta.to(device)*expert_value
        elif ((self.beta == 0.) or (self.critic_key=='agent')) and not (self.critic_key=='expert'):
            value = agent_value
        elif self.beta == 1. or (self.critic_key=='expert'):
            value = expert_value
        else:
            raise Exception('')

        # action = inputs['fixed_expert_actions']
        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs=None, masks=None, device='cuda:1'):

        # get agent info
        if self.sampling_dist == 'agent':
            agent_value, _, _ = self.base(inputs, rnn_hxs, masks)
        elif self.sampling_dist == 'expert':
            agent_value, _, _ = self.base.expert_forward(inputs, rnn_hxs, masks)
        else:
            raise Exception('self.sampling_dist: '+self.sampling_dist)

        # get expert info
        if self.use_learned_expert:
            expert_value, _, _ = self.base.expert_forward(inputs, rnn_hxs, masks)
        else:
            expert_value = None

        # now combine value functions
        if (expert_value is not None) and (self.critic_key=='mixture'):
            value = (1-self.beta.to(device))*agent_value + self.beta.to(device)*expert_value
        elif ((self.beta == 0.) or (self.critic_key=='agent')) and not (self.critic_key=='expert'):
            value = agent_value
        elif self.beta == 1. or (self.critic_key=='expert'):
            value = expert_value
        else:
            raise Exception('')

        # return value
        return value

    def evaluate_actions(self, inputs, action, rnn_hxs=None, masks=None, detach_encoder=False, device='cuda:1'):

        if self.sampling_dist == 'agent':
            agent_value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
            agent_dist = self.agent_dist(actor_features)
        else:
            agent_value, actor_features, rnn_hxs = self.base.expert_forward(inputs, rnn_hxs, masks)
            agent_dist = self.expert_dist(actor_features)

        # get expert info
        if self.use_learned_expert:
            expert_value, actor_features, rnn_hxs = self.base.expert_forward(inputs, rnn_hxs, masks)
            expert_dist = self.expert_dist(actor_features)
        else:
            expert_value = None
            expert_dist = self.fixed_expert(inputs['fixed_expert_actions'].detach(), device)

        # compute scaled beta for normalization
        if (self.beta > 0.) and (self.beta < 1.):
            agent_log_probs = agent_dist.log_probs(action) + torch.log(1-self.beta.to(device))
            expert_log_probs = (expert_dist.log_probs(action) + torch.log(self.beta.to(device)))
            log_probs = torch.cat([agent_log_probs, expert_log_probs],dim=1)
            action_log_probs = torch.logsumexp(log_probs,1).unsqueeze(1)
        elif self.beta == 0.:
            action_log_probs = agent_dist.log_probs(action)
        elif self.beta == 1. or force_expert_sample:
            action_log_probs = expert_dist.log_probs(action)
        else:
            raise Exception('something is guffed.')

        # now combine value functions
        if (expert_value is not None) and (self.critic_key=='mixture'):
            value = (1-self.beta.to(device))*agent_value + self.beta.to(device)*expert_value
        elif ((self.beta == 0.) or (self.critic_key=='agent')) and not (self.critic_key=='expert'):
            value = agent_value
        elif self.beta == 1. or (self.critic_key=='expert'):
            value = expert_value
        else:
            raise Exception('')

        # get agents dist entropy
        if (self.beta > 0.) and (self.beta < 1.):
            agent_entropy = agent_dist.entropy().mean()
            expert_entropy = expert_dist.entropy().mean()
            dist_entropy = (1-self.beta.to(device))*agent_entropy + self.beta.to(device)*expert_entropy
        elif self.beta == 0.:
            dist_entropy = agent_dist.entropy().mean()
        elif self.beta == 1. or force_expert_sample:
            dist_entropy = expert_dist.entropy().mean()
        else:
            raise Exception('something is guffed.')

        # return info
        return value, action_log_probs, dist_entropy, rnn_hxs

    def evaluate_expert_actions(self, inputs, action, rnn_hxs=None, masks=None, detach_encoder=False, device='cuda:1'):

        # get expert info
        if self.use_learned_expert:
            value, actor_features, rnn_hxs = self.base.expert_forward(inputs, rnn_hxs, masks, detach_encoder)
            expert_dist = self.expert_dist(actor_features)
        else:
            value = None
            expert_dist = self.fixed_expert(inputs['fixed_expert_actions'], device)
        # compute expert log-probs
        expert_log_probs = expert_dist.log_probs(action)
        # compute the full log-prob
        dist_entropy = expert_dist.entropy().mean()
        # return info
        return value, expert_log_probs, dist_entropy, rnn_hxs

    def evaluate_agent_actions(self, inputs, action, rnn_hxs=None, masks=None, detach_encoder=False, device='cuda:1'):
        # get agent info
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, detach_encoder)
        agent_dist = self.agent_dist(actor_features)
        # compute expert log-probs
        agent_log_probs = agent_dist.log_probs(action)
        # compute the full log-prob
        dist_entropy = agent_dist.entropy().mean()
        # return info
        return value, agent_log_probs, dist_entropy, rnn_hxs

    def agent_expert_divergence(self, inputs):
        # get features
        _, expert_features, _ = self.base.expert_forward(inputs).detach()
        _, agent_features, _ = self.base.forward(inputs).detach()
        # create dist objects
        expert_dist_ = self.expert_dist(expert_features)
        agent_dist_ = self.agent_dist(agent_features)
        # compute kl div
        div = torch.distributions.kl.kl_divergence(expert_dist_, agent_dist_)
        # return it
        return div

class MLP(nn.Module):

    """Convolutional encoder for image-based observations."""
    def __init__(self, input_dim, output_dim, hidden_size=64, hidden_depth=2, activ=nn.Tanh, output_mod=None, init_func=None):
        super().__init__()

        if init_func is None:
            init_func = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0))
        if hidden_depth == 0:
            mods = [init_func(nn.Linear(input_dim, output_dim))]
        else:
            mods = [init_func(nn.Linear(input_dim, hidden_size)), activ()]
            for i in range(hidden_depth - 1):
                mods += [init_func(nn.Linear(hidden_size, hidden_size)), activ()]
            mods.append(init_func(nn.Linear(hidden_size, output_dim)))
        if output_mod is not None:
            mods.append(output_mod)
        self.main = nn.Sequential(*mods)

    def forward(self, obs, detach=False):
        out = self.main(obs)
        if detach:
            out = out.detach()
        return out

class Flatten(nn.Module):
    """ Flattens output, can be used in specific layers below. """
    def forward(self, x):
        return x.view(x.size(0), -1)

class PixelEncoder(nn.Module):

    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, num_layers=2, num_filters=32,
                     feature_dim=50, stride=(2,2),  kernel_size=(3,3), output_logits=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.output_logits = output_logits
        # try 2 5x5s with strides 2x2. with samep adding, it should reduce 84 to 21, so with valid, it should be even smaller than 21.
        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, kernel_size, stride=stride)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, kernel_size, stride=stride))
        # compute the output size
        try:
            fake_input = torch.zeros(obs_shape).unsqueeze(0)
        except:
            obs_shape = tuple(obs_shape)
            fake_input = torch.zeros(obs_shape).unsqueeze(0)
        conv = torch.relu(self.convs[0](fake_input))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
        out_dim = conv.view(conv.size(0), -1).size()[1]
        print('PixelEncoder outputsize:', out_dim)
        # set the output layers
        self.fc = nn.Linear(out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward_conv(self, obs):
        if len(obs.size()) > len(self.obs_shape)+1:
            obs = obs.squeeze(0)
        conv = torch.relu(self.convs[0](obs))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()
        out = self.fc(h)
        out = self.ln(out)
        out = torch.tanh(out)
        return out

class NNBase(nn.Module):
    """ Under Construction ...."""
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)
        return x, hxs

class CARLABase(NNBase):
    """
    Nueral network based used in carla experiments. Contains the models for all
    inputes including experts, encoders, and agent parameters.
    """
    def __init__(self, num_inputs, params, recurrent=False, hidden_size=64, hidden_depth_=2):
        super(CARLABase, self).__init__(recurrent, num_inputs, hidden_size)

        # set critic inits
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        # set frame-based encoder
        frame_obs_space = num_inputs['frames'].shape

        self.agent_encoder = PixelEncoder(frame_obs_space,
            num_filters = params.agent_numfilters, stride = params.agent_stride,
            kernel_size = params.agent_kernelsize, feature_dim = params.agent_featuredim,
            num_layers = params.agent_convlayers)

        # set bird-view encoder
        self.use_compressed_state = params.use_compressed_state
        self.observation_space = num_inputs

        if not self.use_compressed_state:
            self.expert_obs_space = sum(num_inputs['birdview'].shape)
            self.expert_encoder = PixelEncoder(num_inputs['birdview'].shape,
                num_filters = params.expert_numfilters, stride = params.expert_stride,
                kernel_size = params.expert_kernelsize, feature_dim = params.expert_featuredim,
                num_layers = params.expert_convlayers)
            self.expert_output = self.expert_encoder(torch.zeros(num_inputs['birdview'].shape).unsqueeze(0)).detach()
        else:
            self.expert_obs_space = num_inputs['compact_vector'].shape[0]*num_inputs['compact_vector'].shape[1]
            self.expert_encoder = nn.Identity()
            self.expert_encoder_placeholder = nn.Linear(1, 1)
            self.expert_output = self.expert_encoder(torch.zeros(self.expert_obs_space).unsqueeze(0)).detach()

        # compute the output size of encoders
        self.agent_output = self.agent_encoder(torch.zeros(frame_obs_space).unsqueeze(0)).detach()

        # compute speed info
        self.speed_dim = torch.tensor(num_inputs['speeds'].shape).prod().item()

        # compute prev-action input
        self.prevaction_dim = torch.tensor(num_inputs['prev_actions'].shape).prod().item()

        # compute one-hot-commands info
        self.total_commands = 4
        self.command_dim = num_inputs['commands'].shape[0]*self.total_commands

        # compute the total input dim
        self.expert_input_dim = self.expert_output.reshape(-1).size()[0] + self.command_dim + self.speed_dim + self.prevaction_dim
        self.agent_input_dim = self.agent_output.reshape(-1).size()[0] + self.command_dim + self.speed_dim + self.prevaction_dim

        # add in an MLP controller
        expert_hidden_size = params.expert_hidden_size
        agent_hidden_size = params.agent_hidden_size

        #
        self.agent = MLP(self.agent_input_dim, output_dim=hidden_size, hidden_size=agent_hidden_size,
                    hidden_depth = params.agent_mlplayers, activ=nn.ReLU, output_mod=nn.ReLU())
        # create differentiable expert
        self.expert = MLP(self.expert_input_dim, output_dim=hidden_size, hidden_size=expert_hidden_size,
                    hidden_depth = params.expert_mlplayers, activ=nn.ReLU, output_mod=nn.ReLU())

        # add in a critic base mlp
        self.agent_critic = nn.Sequential(
            init_(nn.Linear(self.agent_input_dim+1, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        self.agent_critic_linear = nn.Linear(hidden_size, 1)

        # expert critic base mlp
        self.expert_critic = nn.Sequential(
            init_(nn.Linear(self.expert_input_dim+1, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())
        self.expert_critic_linear = init_(nn.Linear(hidden_size, 1))

        #
        self.agent_output_size = hidden_size
        self.expert_output_size = hidden_size

        # call train
        self.train()

    def one_hot(self, labels):
        z = torch.nn.functional.one_hot(labels.long(),self.total_commands)
        return z.reshape(z.size()[0], torch.prod(torch.tensor(z.size()[1:])))

    def expert_forward(self, inputs, rnn_hxs=None, masks=None, detach_encoder=False):

        if len(inputs['speeds'].size())==1:
            commands = self.one_hot(inputs['commands']).float().unsqueeze(0)
            speeds = inputs['speeds'].unsqueeze(0)
            timestep = inputs['time'].float().unsqueeze(0)
        else:
            commands = self.one_hot(inputs['commands']).float()
            speeds = inputs['speeds']
            timestep = inputs['time'].float()

        # use compact state ?
        if self.use_compressed_state:
            birdview_encoding = self.expert_encoder(inputs['compact_vector'].reshape(speeds.size()[0],-1))
        else:
            birdview_encoding = self.expert_encoder(inputs['birdview'])

        # do we want to detach the encoder
        if detach_encoder:
            birdview_encoding = birdview_encoding.detach()

        #
        prev_actions = inputs['prev_actions'].reshape(speeds.size()[0],-1).float()

        # combine
        policy_input = torch.cat([birdview_encoding, commands, speeds, prev_actions], dim=1)
        vf_input = torch.cat([birdview_encoding, commands, speeds, prev_actions, timestep], dim=1)

        # pass through policy
        x = self.expert(policy_input)

        # pass through value function
        x_vf = self.expert_critic(vf_input)
        x_vf = self.expert_critic_linear(x_vf)

        # return it all
        return x_vf, x, rnn_hxs

    def forward(self, inputs, rnn_hxs=None, masks=None, detach_encoder=False):

        # get encoding
        frontview_encoding = self.agent_encoder(inputs['frames'])

        # do we want to detach the encoder
        if detach_encoder:
            frontview_encoding = frontview_encoding.detach()

        # the other inputs
        commands = self.one_hot(inputs['commands']).float()
        speeds = inputs['speeds']
        timestep = inputs['time'].float()
        prev_actions = inputs['prev_actions'].reshape(speeds.size()[0], self.prevaction_dim).float()

        # combine
        policy_input = torch.cat([frontview_encoding, commands, speeds, prev_actions], dim=1)
        vf_input = torch.cat([frontview_encoding, commands, speeds, prev_actions, timestep], dim=1)

        # pass through policy
        x = self.agent(policy_input)

        # pass through value function
        x_vf = self.agent_critic(vf_input)
        x_vf = self.agent_critic_linear(x_vf)

        # return it all
        return x_vf, x, rnn_hxs
