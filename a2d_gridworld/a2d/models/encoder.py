# This code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Licence.  The full licence and information is available at:
# https://github.com/plai-group/a2d/blob/master/LICENCE.md
# © Andrew Warrington, J. Wilder Lavington, Adam Scibior, Mark Schmidt & Frank Wood.
# Accompanies the paper: Robust Asymmetric Learning in POMDPs, ICML 2021.

# Imports.
import torch
import torch.nn as nn


def weight_init(m):
    """
    Custom weight init for Conv2D and Linear layers.
    :param m:
    :return:
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    """
    Define a standard MLP.
    :param input_dim:
    :param hidden_dim:
    :param output_dim:
    :param hidden_depth:
    :param output_mod:
    :return:
    """
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


class PixelEncoder(nn.Module):
    """
    Convolutional encoder for image-based observations.
    """

    def __init__(self, obs_shape, num_layers=4, num_filters=32, feature_dim=64, output_logits=False):
        """
        Define the encoder.  Outputs a flat vector of size feature_dim.
        :param obs_shape:
        :param num_layers:
        :param num_filters:
        :param feature_dim:
        :param output_logits:
        """
        super().__init__()

        self.DTYPE = torch.double

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        # Run forward with some dummy inputs to work out the size of the tensors required.
        # Try 2 5x5s with strides 2x2. with ``same'' padding.
        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        if torch.is_tensor(obs_shape):
            fake_input = torch.zeros(*obs_shape).unsqueeze(0)
        else:
            obs_shape = tuple(obs_shape)
            fake_input = torch.zeros(obs_shape).unsqueeze(0)
        conv = torch.relu(self.convs[0](fake_input))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
        out_dim = conv.view(conv.size(0), -1).size()[1]

        # set the output layers
        self.fc = nn.Linear(out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

        self.to(self.DTYPE)

        # This will allow us to freeze the gradients if need be.
        self.PRETRAINED = False


    def forward_conv(self, obs):
        """
        Call the convolutional layers.
        :param obs:
        :return:
        """
        # Fix type and normalize.
        obs = obs.type(next(self.parameters()).type())
        if obs.max() > 1.:
            obs = obs / 255.

        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h


    def forward(self, obs, detach=False):
        """
        Run the encoder.  Calls the convolutional layers and then the output layers.
        :param obs:
        :param detach:
        :return:
        """

        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out


class Identity(nn.Module):
    """
    AW - identity "encoder".  Means that if we are using flat representations we can
    still define an object called encoder.  This means that pytorch can generate
    an empty parameter tensor when required.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, _x, detach=False):
        return _x
