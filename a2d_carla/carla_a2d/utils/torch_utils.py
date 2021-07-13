
# general
import numpy as np

# torch stuff
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def copy_params(target, source):
    """ copies nueral network parameters between to networks. """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def save_network(path, model):
    """ saves models. """
    # check that the directory exists
    if not os.path.exists(path):
        os.makedirs(path)
    # now direct save using pytorch built in
    torch.save(model.state_dict(), path)
    # nothing to return
    return None

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    """ outdated. """
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

def get_n_params(model):
    """ returns the number of parameters. """
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def check_model_weights(model1, model2):
    """ checks if the model weights are the same. """
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

# Get a render function
def get_render_func(venv):
    """ outdated. """
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    """ outdated. """
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None

def init(module, weight_init, bias_init, gain=1):
    """ Used int mlp initialization. """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
