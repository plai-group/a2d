# This code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Licence.  The full licence and information is available at:
# https://github.com/plai-group/a2d/blob/master/LICENCE.md
# © Andrew Warrington, J. Wilder Lavington, Adam Scibior, Mark Schmidt & Frank Wood.
# Accompanies the paper: Robust Asymmetric Learning in POMDPs, ICML 2021.

import torch as ch
from torch.distributions.categorical import Categorical
import numpy as np

CKPTS_TABLE = 'checkpoints'

def cu_tensorize(t):

    return ch.tensor(t).float().cuda()

def cpu_tensorize(t):

    return ch.tensor(t).float()

def gpu_mapper():
    return ch.device('cuda:0') if not cpu else ch.device('cpu')

def shape_equal_cmp(*args):

    for i in range(len(args)-1):
        if args[i].shape != args[i+1].shape:
            s = "\n".join([str(x.shape) for x in args])
            raise ValueError("Expected equal shapes. Got:\n%s" % s)
    return True

def shape_equal(a, *args):

    for arg in args:
        if list(arg.shape) != list(a):
            if len(arg.shape) != len(a):
                raise ValueError("Expected shape: %s, Got shape %s" \
                                    % (str(a), str(arg.shape)))
            for i in range(len(arg.shape)):
                if a[i] == -1 or a[i] == arg.shape[i]:
                    continue
                raise ValueError("Expected shape: %s, Got shape %s" \
                                    % (str(a), str(arg.shape)))
    return shape_equal_cmp(*args)

def scat(a, b, axis):
    if a is None:
        return b
    return ch.cat((a, b), axis)

def determinant(mat):
    return ch.exp(ch.log(mat).sum())

def safe_op_or_neg_one(maybe_empty, op):
    if maybe_empty.nelement() == 0:
        return -1.
    else:
        return op(maybe_empty)

def discount_path(path, h):
    curr = 0
    rets = []
    for i in range(len(path)):
        curr = curr*h + path[-1-i]
        rets.append(curr)
    rets =  ch.stack(list(reversed(rets)), 0)
    return rets

def get_path_indices(not_dones):
    indices = []
    num_timesteps = not_dones.shape[1]
    for actor in range(not_dones.shape[0]):
        last_index = 0
        for i in range(num_timesteps):
            if not_dones[actor, i] == 0.:
                indices.append((actor, last_index, i + 1))
                last_index = i + 1
        if last_index != num_timesteps:
            indices.append((actor, last_index, num_timesteps))
    return indices

def select_prob_dists(pds, selected=None, detach=True):
    if type(pds) is tuple:
        if selected is not None:
            tup = (pds[0][selected], pds[1])
        else:
            tup = pds
        return tuple(x.detach() if detach else x for x in tup)
    out = pds[selected] if selected is not None else pds
    return out.detach() if detach else out


def vjp(f_x, theta, v, create=True):
    '''
    Vector-jacobian product
    Calculates v^TJ, or J^T v, using standard backprop
    Input:
    - f_x, function of which we want the Jacobian
    - theta, variable with respect to which we want Jacobian
    - v, vector that we want multiplied by the Jacobian
    Returns:
    - J^T @ v, without using n^2 space
    '''
    grad_list = ch.autograd.grad(f_x, theta, v, retain_graph=True, create_graph=create)
    return ch.nn.utils.parameters_to_vector(grad_list)

def jvp(f_x, theta, v):
    '''
    Jacobian-vector product
    Calculate the Jacobian-vector product, see
    https://j-towns.github.io/2017/06/12/A-new-trick.html for math
    Input:
    - f_x, function of which we want the Jacobian
    - theta, variable with respect to which we want Jacobian
    - v, vector that we want multiplied by the Jacobian
    Returns:
    - J @ v, without using n^2 space
    '''
    w = ch.ones_like(f_x, requires_grad=True)
    JTw = vjp(f_x, theta, w)
    return vjp(JTw, w, v)

def cg_solve(fvp_func, b, nsteps):
    # Initialize the solution, residual, direction vectors
    x = ch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    new_rnorm = ch.dot(r,r)
    for _ in range(nsteps):
        rnorm = new_rnorm
        fvp = fvp_func(p)
        alpha = rnorm / ch.dot(p, fvp)
        x += alpha * p
        r -= alpha * fvp
        new_rnorm = ch.dot(r, r)
        ratio = new_rnorm / rnorm
        p = r + ratio * p
    return x

def backtracking_line_search(f, x, expected_improve_rate,
                             num_tries=10, accept_ratio=.1, verbose=False):  # , _net=None, _pds=None, _test_pds=None):
    # f gives improvement
    for i in range(num_tries):
        scaling = 2**(-i)
        scaled = x * scaling
        improve = f(scaled)
        # if improve == -float('inf'):
        #     p = 0
        #     _net.calc_kl(_pds, _test_pds).mean()
        expected_improve = expected_improve_rate * scaling
        if improve/expected_improve > accept_ratio and improve > 0:
            if verbose:
                print("We good! %f" % (scaling,))
            return scaled
    return 0.


def _check_param_device(param, old_param_device):
    r""" MODIFIED FROM torch.nn.utils -> _check_param_device

    This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.

    Arguments:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.

    Returns:
        old_param_device (int): report device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device

def assign(vec, parameters):
    r"""MODIFIED FROM torch.nn.utils -> vector_to_parameters

    Convert one vector to the parameters

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, ch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(ch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Dont apply to parameters that dont require gradients.
        if not param.requires_grad:
            continue

        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param

