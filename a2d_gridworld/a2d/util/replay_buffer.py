# This code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Licence.  The full licence and information is available at:
# https://github.com/plai-group/a2d/blob/master/LICENCE.md
# © Andrew Warrington, J. Wilder Lavington, Adam Scibior, Mark Schmidt & Frank Wood.
# Accompanies the paper: Robust Asymmetric Learning in POMDPs, ICML 2021.

import torch
import gym
import torchvision

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from collections import namedtuple
from copy import deepcopy
from pprint import pprint

STATE_DTYPE = torch.double
OBS_DTYPE = torch.half


class ReplayBuffer():
    """
    AW - define a fairly modular replay buffer for storing true world states paired to observed values.

    NOTE - this class got pulled apart a bit, and so some of the frame stacking isn't guaranteed to work
    properly for more complex observations.  I'm in the process of re-making this and making it better.
    """

    def __init__(self, default_minibatch_size, _buffer_max_size, _augmentate=None, _USE_EPOCH_SAMPLE=True,
                 _STATE_DTYPE=STATE_DTYPE, _OBS_DTYPE=OBS_DTYPE):
        """
        AW - initilize the buffer.  The buffer itself is created on the first call to push, as the shapes can be
        extracted then.  The buffer itself is implemented using a circular counter inscribing new data so that it
        overwrites the oldest data.

        Data can be accessed in two ways, selected using the _USE_EPOCH_SAMPLE constant.
        Epoch sampling is the default where data is looped over.  Random sampling is the fallback where the buffer
        is randomly accessed.  Basic buffer info is accessible via the property Buffer.buffer_info.  Every time
        the buffer is pushed to the epoch shuffle is reset and the counter restarts.

        Buffer.total_sampled records the total number of samples that have been accessed, and Buffer.examples_seen
        tracks the total number of samples that have been pushed to this buffer during its lifetime.

        Two reset options exist.  Buffer.wipe clears the current data in a lazy way (resetting the counters and lengths)
        whereas Buffer.hard_reset re-initializes the whole buffer.  Hard resetting the buffer clears any persistent
        counters and so is probably not especially safe...  Wipe preserves these counters.

        The datatype the data is stored under is defined by the global variables defined at the top of this script,
        however, they can be overwritten by passing different datatypes are arguments to this initializer.

        Can pass in some image augmentation options through augmentator.  Leaving it to default as None means the
        default augmentation is used.  A list can be passed in to define which augmentations should be applied.

        Frame stacks are dealt with by constructing an intermediate matrix (id_pairing) that tracks the frames that
        are correlated with the ith state.  This matrix is updated on push.

        Image samples have shape (N, S, W, H, C) or (Samples, FrameStack, Width, Height, Channels(3)).

        :param: buffer_max_size - maximum number of entries in the FIFO buffer.
        :param: augmentator=None - can optionally pass in a list of image augmentations to apply.  Look in
                                    Buffer.augment to get the naming convention for the augmentations.
        :param: USE_EPOCH_SAMPLE=True - when sampling, cycle through data in epoch style (Default=True).
        :param: _STATE_DTYPE=STATE_DTYPE - default datatype for storing states.
        :param: _OBS_DTYP=_OBS_DTYPE - default datatype for storing observations.
        """

        # Set the default minibatch size.  Can be changed later on.
        self.default_minibatch_size = default_minibatch_size

        # Inscribe args.
        self._augmentate = _augmentate

        # Inscribe the length of the buffer.
        self.buffer_max_size = _buffer_max_size

        # Define sampling mechanism.
        self.use_epoch_sample = _USE_EPOCH_SAMPLE

        # These will be overwritten on the first push.
        self.states = None                            # State store.
        self.obs = None                               # Obs store.
        self.state_shape = None                       # Shape of state (no stacking).
        self.obs_shape = None                         # Shape of raw obs (without stack)
        self.frame_stack_size = None                  # Frames to stack
        self.augmentator = _augmentate                # To be overwritten if images are inscribed later.
        self._do_augmentate = None

        # Define the datatype to use to store each field.
        self.state_type = _STATE_DTYPE
        self.obs_type = _OBS_DTYPE

        # Define misc stuff.
        self.active_length = 0                        # Store the current length of the buffer.
        self.counter = 0                              # Store next element to be accessed.
        self.examples_seen = 0                        # Count the number of examples that have flown through.
        self.epoch_sample_counter = 0                 # If we are going to sample the replay buffer in an epoch-style.
        self.epoch_sample_order = torch.empty((1,))   # Overwritten at every push and partially consumed at each sample.
        self.epoch_counter = 0                        # Store how many epochs on current buffer.
        self.total_sampled = 0                        # Count the total number of samples, sampled...!
        self.epoch_batch_counter = 0                  # Count how many batches have been generated.  Volatile.

    def __len__(self):
        """
        AW - return the active length of the buffer.
        :return:
        """
        return self.active_length
    
    def __iter__(self):
        """
        AW - define the iteration properties of the buffer.  Sicne we have __next__ implemented this class is an iter.
        :return: return signature of sample.
        """
        n_batch = np.floor(len(self) / self.default_minibatch_size)
        return iter(self.__next__, 'STOP')

    def __next__(self):
        """
        AW - functionality of enumerate.
        :return:
        """
        current_epoch = deepcopy(self.epoch_counter)
        self.epoch_batch_counter += 1
        if (self._do_augmentate is True) or (self._do_augmentate is None):
            batch = self.sample_and_augment()
        else:
            batch = self.sample()
        new_epoch = deepcopy(self.epoch_counter)
        if current_epoch == new_epoch:
            return batch
        else:
            return 'STOP'  # Send back Sentinel signal.
    

    """ Define core functionality. --------------------------------------------------------------------------------- """

    def push(self, data, expert_action=None):
        """
        AW - push data to the buffer.  Data is assumed to be in the form of a tuple.

        Assumes observation frames come in in chronological order, i.e. is a FIFO buffer.

        # TODO - process expert action tensor as well.

        :param: data - (state(np), observation(tensor), terminals(iterable)), all equal length.
        :returns: None
        """
        # Pre-process the incoming data.
        if type(data) != tuple:
            st = np.asarray(data.state)
            te = 1-np.asarray(data.mask)  # Mask is whether it is _not_ terminal.
            ob = torch.stack(data.stored_observe)

            # Pack these into a tuple.
            data = (st, ob, te)

        assert len(data[0]) == len(data[1]) == len(data[2]), 'State and observation need to be the same length.'
        n_to_add = min((len(data[0]), self.buffer_max_size))
        states = data[0]
        obs = data[1]
        terminals = data[2]

        # Convert to tensors and change type, if need be.
        states = states if torch.is_tensor(states) else torch.tensor(states)
        obs = obs if torch.is_tensor(obs) else torch.stack(obs)
        states = states.type(self.state_type)
        obs = obs.type(self.obs_type)

        # Limit the length of the states and observations.
        states = states[:n_to_add]
        obs = obs[:n_to_add]

        # Decide if we are going to fill the buffer...
        buffer_top = np.min((self.counter + n_to_add, self.buffer_max_size))
        states_top = np.min((n_to_add, self.buffer_max_size))

        # If this is the first push, define the buffer and grab the sizes.
        if self.states is None:
            assert states_top == buffer_top  # Need both of these to be the same size.
            self.active_length = states_top

            # Build out frame stacking indexer.
            self.frame_stack_size = obs.shape[1]    # The obs has some shape and the first dim is the stack.

            # Construct state buffer.
            self.counter = states_top
            self.state_shape = states.shape[1:]
            self.obs_shape = obs.shape[1:]  # IF USING id_pairing: [2:]
            self.states = torch.zeros([self.buffer_max_size] + list(self.state_shape))
            self.obs = torch.zeros([self.buffer_max_size] + list(self.obs_shape))

            # Might as well do a clean inscribe here.
            self.states[0:buffer_top] = deepcopy(states[0:states_top])

            # IF USING id_pairing: obs[0:states_top]  # The zeroth entry contains all required frames.
            self.obs[0:buffer_top] = deepcopy(obs[0:states_top])

            # Update the augmentator.
            if len(self.obs.shape) != 5: self.augmentator = False       # Must be NxSxRGB images.
            # problem_here_should_be_five for images. TODO
            if self.obs.shape[-1] != 3: self.augmentator = False        # Final channel must be RGB channels.

            # Build the augmentator now we have the size of the observations.
            self._build_augmentator()

        else:

            # We have previously pushed to the buffer, so now we just need to update it.
            n_top = min((self.buffer_max_size - self.counter), n_to_add)
            self.states[self.counter:(self.counter + n_top)] = deepcopy(states[:n_top])
            self.obs[self.counter:(self.counter + n_top)] = deepcopy(obs[:n_top])  # IF using id_pairing [.. , 0]
            # self.id_pairing[self.counter:(self.counter + n_top)] = deepcopy(temp_pair_id[:n_top])

            # Update the active length
            self.active_length = np.min((self.active_length + n_to_add, self.buffer_max_size))

            # There were some leftover samples so add them to the bottom.
            if n_top != n_to_add:
                self.counter = len(states) - n_top  # Hard set counter.
                self.states[:self.counter] = deepcopy(states[n_top:])
                self.obs[:self.counter] = deepcopy(obs[n_top:])  # IF using id_pairing [.. , 0]
                # self.id_pairing[:self.counter] = deepcopy(temp_pair_id[n_top:]) % self.buffer_max_size

            else:
                # Else update/roll counter.
                self.counter = (self.counter + n_to_add) % self.buffer_max_size

        # Update examples seen.
        self.examples_seen += n_to_add

        # Update the sampling order.
        self._update_epoch_sampling_order()

        # This is the first epoch of the updated buffer.
        self.epoch_counter = 0
        self.epoch_sample_counter = 0
        self.epoch_batch_counter = 0

    def sample(self, n_samp=None, _flatten=True):
        """
        AW - sample a minibatch according to the prescribed method.  Default is epoch-style sampling.
        :param n_samp: length of minibatch to sample.
        :return: return signature of respective called functions.  (Probably (state, obs) tensor tuple.
        """
        if n_samp is None:
            n_samp = self.default_minibatch_size
        
        self.total_sampled += n_samp
        if self.use_epoch_sample:
            sample = self._get_epoch_minibatch(n_samp)
        else:
            sample = self._random_sample(n_samp)

        return sample

    def sample_and_augment(self, n_samp=None, _flatten=True):
        """
        AW - wrapper method for sampling a minibatch and then augmenting.
        :param n_samp: int.  number of samples to sample.
        :return: return signature of respective called functions.  (Probably (state, obs) tensor tuple.
        """
        if n_samp is None:
            n_samp = self.default_minibatch_size
            
        samples = self.sample(n_samp, _flatten=False)
        samples = self.augment(samples)

        return samples

    def augment(self, data):
        """
        AW - augment the sampled observations.  Requires that the image is a three-channel RGB image.
        :param data: tuple of (state, observation) as returned by Buffer.sample
        :return: tuple of (state, observation) as returned by Buffer.sample, but with augmented images.
        """

        # Check that we are applying an augmentation
        if not self.augmentator:
            return data

        # Check the observation is the right size.
        if 'identity' not in str(self.augmentator):
            assert self.obs.shape[-1] == 3, 'Final channel must be RGB image.'

        obs_aug = self.augmentator(data[1])
        data_new = (data[0], obs_aug)
        return data_new

    def wipe(self):
        """
        AW - clear the buffer.  NOTE - currently lazy clear, done by setting active length and counter to zero.
        :return: None
        """
        self.counter = 0
        self.active_length = 0

    def hard_reset(self):
        """
        AW - fully reset buffer.  Not a massive amount of point in calling this, just delete the buffer and
        define a new one and GC will take care of the rest.
        :return: None
        """
        print('[ReplayBuffer]: Warning: Hard resetting buffer...')
        self.__init__(self.default_minibatch_size, self.buffer_max_size, self._augmentate)
        self.wipe()
        self.examples_seen = 0


    """ Define hidden functions -- not to be accessed from outside this class. --------------------------------------- """


    def _unflatten_rgb(self, _ob):
        ob = _ob.reshape((_ob.shape[0], 3, -1, _ob.shape[2], _ob.shape[3])).permute(0, 2, 3, 4, 1)
        return ob


    def _random_sample(self, n_samp):
        """
        AW - sample from the replay buffer in a random fashion -- Introduces additional variance in the binomial
        nature of this sampling scheme.
        :param: n_samp - number of samples to draw from buffer.
        :returns: tuple(state, obs) - equal length torch tensors.
        """
        idx = np.random.randint(0, self.active_length, (n_samp,))

        # # Grab the pairing ids.
        # pairing_ids = self.id_pairing[idx].long()
        pairing_ids = torch.tensor(idx)

        # Grab the states -- no stacking required, so take first entry.
        state = self.states[pairing_ids[:, 0].long()].clone()

        # Go in and get correct observations and stack them.
        obs = self.obs[pairing_ids.long()].clone()

        return state, obs

    def _update_epoch_sampling_order(self):
        """
        AW - redraw the new sampling order.
        :return:
        """
        self.epoch_sample_order = np.arange(self.active_length)
        np.random.shuffle(self.epoch_sample_order)
        self.epoch_sample_order = torch.tensor(self.epoch_sample_order).type(torch.int32)

    def _get_epoch_minibatch(self, n_samp):
        """
        AW - get the next minibatch in the epoch.
        :param n_samp: number of samples to obtain
        :return: tuple(state, observation) - torch tensors of state and observation in minibatch.
        """
        assert n_samp < self.active_length, "Can't do epoch-style sampling requesting more than the length of buffer."

        # If we won't complete the epoch in this sample.
        if n_samp < len(self.epoch_sample_order):
            # self.epoch_sample_order = torch.arange(len(self.epoch_sample_order))
            idx = self.epoch_sample_order[:n_samp]
            self.epoch_sample_order = self.epoch_sample_order[n_samp:]
        else:
            old_idx = deepcopy(self.epoch_sample_order)
            self._update_epoch_sampling_order()
            new_idx = self.epoch_sample_order[:(n_samp-len(old_idx))]
            self.epoch_sample_order = self.epoch_sample_order[len(old_idx):]
            idx = torch.cat((old_idx, new_idx))

            # Completed another minibatch.
            self.epoch_counter += 1             # Next epoch.
            self.epoch_batch_counter = 0        # First batch.

        # Increment volatile epoch counter.
        self.epoch_sample_counter += len(idx)

        # # Grab the pairing ids.
        # pairing_ids = self.id_pairing[idx.long()]

        # Grab the states -- no stacking required, so take first entry.
        pairing_ids = idx.long().unsqueeze(-1)
        state = self.states[pairing_ids[:, 0].long()].clone()

        # Go in and get correct observations and stack them.
        pairing_ids = idx.long()
        obs = self.obs[pairing_ids.long()].clone()

        return state, obs

    def _build_augmentator(self):
        """
        AW - Build the augmentation function.  Uses the Buffer._augmentate field.
        :return: None (side effect of Buffer.augmentator being a function)
        """

        # If augmentator is set to false there is no augmentation to be used (obs probably isn't an image).
        if self.augmentator is False:
            self.augmentator = self._augment_identity
            print('[Augmentor]: Not augmentating data.')
            return None

        # Holder for the augmentator functions.
        aug = []

        # Build the list of augmentations.
        if type(self.augmentator) is not list:
            augmentations = ['speckle', 'crop']
        else:
            augmentations = self.augmentator

        print('[Augmentor]: Using {} augmentations.'.format(augmentations))

        # Crop image.
        if 'crop' in augmentations:
            aug.append(self._augment_crop())

        # Whole-image changes.
        if 'colorjitter' in augmentations:
            aug.append(self._augment_colorjitter())

        # White noise application.
        if 'speckle' in augmentations:
            aug.append(self._augment_speckle())

        # Resize to correct size.
        aug.append(self._augment_resize(self.obs_shape[1], self.obs_shape[2]))

        # Compose and inscribe.
        self.augmentator = torchvision.transforms.Compose(aug)

    @staticmethod
    def _augment_identity(obs):
        """
        AW - No transform/augmentation.
        :param obs:
        :return:
        """
        return obs

    @staticmethod
    def _augment_resize(_w, _h):
        """
        AW - resize the image.
        :param self:
        :param _w: target width.
        :param _h: target height.
        :return:
        """

        class Resize(object):
            """ AW - holder class for resize transformation.  TODO - currently looping..."""

            def __init__(self, _w, _h):
                """ AW - init. """
                self.w = _w
                self.h = _h

            def __call__(self, img):
                """ AW - apply transform. """
                img_out = torch.zeros((img.shape[0], img.shape[1], self.w, self.h, img.shape[4]))
                for _n in range(len(img_out)):
                    img_out[_n] = F.interpolate(img[_n].permute(0, 3, 1, 2), size=(self.w, self.h),
                                                mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                return img_out

        return Resize(_w, _h)

    @staticmethod
    def _augment_colorjitter():
        """
        AW - apply whole-image transformations.

        # TODO - colour jitter isn't currently working. requires conversion to PIL / separate images.

        :return: fucntion to apply augmentation to observation.
        """
        raise NotImplementedError  # Need to manually implement jitter transforms.
        return torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)

    @staticmethod
    def _augment_speckle():
        """
        AW - apply random gaussian noise to image.
        :param data: input observation
        :return: fucntion to apply augmentation to observation.
        """

        class Speckle(object):
            """ AW - holder class for speckle transformation"""

            def __init__(self):
                """ AW - init. """
                self.bounds = [0, 1]
                self.sd = 0.025

            def __call__(self, img):
                """ AW - apply transform. """
                return torch.clamp(img + torch.normal(0, self.sd, size=img.shape), self.bounds[0], self.bounds[1])

        # Return class object.
        return Speckle()

    def _augment_crop(self):
        """
        AW - apply random crop (extraction).  Same crop is applied to all images in the batch.  Would have to resize
        introducing an additional loop otherwise.  Not especially GPU amenable / slow to apply...  Note that we use
        int() and its truncation behaviour to pull image size at least down.

        Currently implemented so that same crop is applied to whole minibatch.  Per-image crops slow down application
        time quite considerably.

        Currently implemented as a functor as so has to be called again to actually apply the obs.

        :param obs: observation tensor.
        :param crop_factor_min: float [0, 1) - minimum relative size to crop to.
        :return: observation with augmentation applied.
        """

        class RandCrop(object):
            """ AW - holder class for random crop transformation. """

            def __init__(self, crop_factor_min=0.9, crop_factor_max=1.1):
                """ AW - init. """
                self.crop_factor_max = crop_factor_max

                # Need to apply additional scaling to compensate for negative crop sizes.
                self.crop_factor_min = crop_factor_min / crop_factor_max

            def __call__(self, img):
                """ AW - apply transform. """
                crop_factor = np.random.uniform(self.crop_factor_min, 1.0, (2,))

                # Resize the image so we can get negative crops.  NOTE - goes backwards in dimensions.
                pad_img = F.pad(img, pad=[0,
                                          0,
                                          np.int(((self.crop_factor_max - 1) / 2) * (img.shape[3])),
                                          np.int(((self.crop_factor_max - 1) / 2) * (img.shape[3])),
                                          np.int(((self.crop_factor_max - 1) / 2) * (img.shape[2])),
                                          np.int(((self.crop_factor_max - 1) / 2) * (img.shape[2])),])

                target_size = torch.tensor((int(pad_img.shape[2] * crop_factor[0]),
                                            int(pad_img.shape[3] * crop_factor[1]))).long()

                origin_1 = torch.tensor(np.random.randint(0, pad_img.shape[2] - target_size[0])).long()
                origin_2 = torch.tensor(np.random.randint(0, pad_img.shape[3] - target_size[1])).long()
                img_aug = pad_img[:, :, origin_1:(origin_1 + target_size[0])]
                img_aug = img_aug[:, :, :, origin_2:(origin_2 + target_size[1])]
                return img_aug

        return RandCrop()


    """ Define menial getters and setters. ------------------------------------------------------------------------- """


    @property
    def get_default_minibatch_size(self):
        return self.default_minibatch_size

    def set_default_minibatch_size(self, _size):
        """
        AW - set the default minibatch size used in enumerate.
        :param _size: New batch size.
        :return: None
        """
        assert type(_size) == int
        self.default_minibatch_size = _size

    def get_expert_numbers(self):
        # TODO - implement this properly.
        return 0, 0

    @property
    def get_buffer_length(self):
        """
        AW - return the length of the buffer.
        :return: int
        """
        return self.active_length

    @property
    def get_examples_seen(self):
        """
        AW - Get the number of examples that have been seen.
        :return: int
        """
        return self.examples_seen

    @property
    def get_is_full(self):
        """
        AW - is the buffer full?
        :return: Bool
        """
        return self.get_buffer_length == self.buffer_max_size

    @property
    def get_state_size(self):
        if self.states is not None:
            return self.states.shape[1:]
        else:
            return (0, )

    @property
    def get_obs_size(self):
        if self.obs is not None:
            return self.obs.shape[1:]
        else:
            return (0, )

    @property
    def samples_proccesed(self):
        return self.examples_seen

    @property
    def buffer_info(self, do_print=True):
        """
        AW - Get general information about the buffer.
        :param: do_print=True - also print buffer info to screen.
        :return: dict - information dictionary.
        """
        info = {'buffer_max_size': self.buffer_max_size,
                'active_length': self.get_buffer_length,
                'if_full': self.get_is_full,
                'examples_seen': self.get_examples_seen,
                'samples_accessed': self.total_sampled,
                'state_size': self.get_state_size,
                'obs_size': self.get_obs_size}
        if do_print: pprint(info)
        return info

    @property
    def epoch_progress(self):
        return float(len(self.epoch_sample_order)) / float(len(self))


class Memory(object):
    """
    # Taken from
    # https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

    Define a really lightweight memory object for storing state tuples.
    """
    def __init__(self):
        self.memory = []
        self.Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(self.Transition(*args))

    def sample(self):
        """Zip up the memory."""
        return self.Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


class Memory_POMDP(object):
    """
    Taken from
    https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

    Define a really lightweight memory object for storing state tuples.
    """
    def __init__(self):
        self.memory = []
        self.Transition_POMDP = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                                          'reward', 'stored_observe'))

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(self.Transition_POMDP(*args))

    def sample(self):
        """Zip up the memory."""
        return self.Transition_POMDP(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


