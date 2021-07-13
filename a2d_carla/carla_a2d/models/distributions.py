import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from carla_a2d.utils.torch_utils import AddBias, init

# Normal
class FixedNormal(torch.distributions.Normal):
    """
    Vectorize multivariate normal distribution (becuase pytorch natively has
    some issues vectorizing these / did when this code was originally made).
    This class inherits everything from .Normal and can use repareterization
    if desired.

    Attributes
    ----------
    ...

    Methods
    -------
    mode()
    returns mean of normal distribution with respect to some action. This is the
    mean action of a given policy using the feature space provided.

    log_probs()
    return the probability of an action conditional on the model of actions given
    states, and the standard deviation defined over each action index.

    entrop()
    return entropy of normal distribution, this should be averaged over a set
    of examples, as the indiviual one only returns this for one example per.

    std()
    returns std of normal distribution with respect to some action. This is the
    std action of a given policy using the feature space provided.
    """

    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean

    def std(self):
        return self.std

class DiagGaussian(nn.Module):
    """
    Something convert a generic feature space to a multivariation Diagnol
    Guassian distribution over actions. This class includes some tweaks that
    allow you to enforce specific types of initial distributions over actions
    like say a mean zero-standard deviation one action distribution (which is
    very much required to use most continuous control on policy methods).

    Attributes
    ----------
    num_inputs: int
    This defines the size of the flattened vector input which defines the output
    feature space from the previous layer.

    num_outputs: int
    Again, this distribution outputs a flat vector of actions, and this attrib
    defines the size of that vector. For our examples this is always size two,
    which is the size of the action space in carla.

    param_sd: [int,None]
    This will determine whether or not the standard deviation will be parameterized
    by some feature space in the same way that the mean is, or if we will not
    actually condition it as is done in many on policy methods like PPO/TRPO/A2C.

    zero_mean: [True,False]
    Indicates if we are to force the output of the linear layer to be mean zero
    standard deviation one. Again initialization is important in RL.

    Methods
    -------
    get_mean(x)
    returns mean of normal distribution with respect to some action. This is the
    mean action of a given policy using the feature space provided.

    get_std(sd_input, min_std)
    returns std of normal distribution with respect to some action. This is the
    std action of a given policy using the feature space provided.

    forward(x, sd_input)
    Returns a fixed normal distribution defined by the mean and sd as defined above.
    To reiterat, this takes a set of features and actually returns a parameterized
    distribution object.
    """

    def __init__(self, num_inputs, num_outputs, param_sd=None, zero_mean=True):
        super(DiagGaussian, self).__init__()
        # initializer
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.
                               constant_(x, 0))
        # set info
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        #
        if zero_mean:
            self.fc_mean.weight.data.mul_(0.0)
            self.fc_mean.bias.data.mul_(0.0)
        # complex
        self.fc_std = nn.Linear(param_sd, num_outputs, bias=True)
        self.fc_std.weight.data.mul_(0.0)
        self.fc_std.bias.data.mul_(0.0)
        # simple
        stdev_init = - 1.25 * torch.ones(num_outputs)
        self.logstd = torch.nn.Parameter(stdev_init)

    def get_mean(self, x):
        action_mean = self.fc_mean(x)
        # action_mean = torch.tanh(action_mean)
        return action_mean

    def get_std(self, sd_input, min_std=1e-12):
        if sd_input is not None:
            action_log_std = self.fc_std(sd_input)
            # action_std = torch.clamp(action_log_std, min=-20., max=2).exp()
            return action_std.exp() + min_std
        else:
            return self.logstd.exp() + min_std

    def forward(self, x, sd_input=None):

        action_mean = self.fc_mean(x)
        # action_mean = torch.tanh(action_mean)

        return FixedNormal(action_mean,  self.get_std(sd_input))
