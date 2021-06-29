# This file was originally released by Chevalier-Boisvert, Maxime and Willems, Lucas and Pal, Suman
# as part of the gym-minigrid environment, released under the Apache 2.0 Licence.  We have
# modified this file, and distribute this modified file as permitted under the terms of the original
# licence.  The modifications are therefore also covered under the CC licence we release under.

# When referencing our work, please also direct citations/references towards the original authors
# of the environment.  Information is available here: https://github.com/maximecb/gym-minigrid.


from a2d.environments.minigrid.gym_minigrid.minigrid import *
from a2d.environments.minigrid.gym_minigrid.register import register

class LavaGapEnv(MiniGridEnv):
    """
    Environment with one wall of lava with a small gap to cross through
    This environment is similar to LavaCrossing but simpler in structure.
    """

    def __init__(self, size, obstacle_type=Lava, seed=None):
        self.obstacle_type = obstacle_type

        # Set the maximum runlength of the env.
        _tmax = 100

        super().__init__(
            grid_size=size,
            max_steps=_tmax,  # int(size*size),
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=seed,
            agent_view_size=9
        )

        # Print out the asymtotic rewards.
        print('TigerDoor: MDP: 6, POMDP: 4')

        # Redefine the observation space to be the compact state.
        _obs = self.reset()
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(_obs), ),
            dtype='uint8'
        )

        # Set up the renderer.
        self.render_type = 'observe'
        self.tile_size = 6

    def _gen_grid(self, width, height, _seed=None):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place barriers.
        _idx = np.arange(width * height).reshape(width, height).T
        _wall_locations = []
        for _l in _wall_locations:
            loc = np.squeeze(np.asarray(np.where(_l == _idx)))
            self.put_obj(Wall(), loc[1], loc[0])

        self.potential_lava_locs = [16, 23, 30,
                                    17, 24, 31,
                                    18, 25, 32]
        self.lava_idx = []
        for _ in range(1):
            self.lava_idx.append(np.random.choice(len(self.potential_lava_locs)))
            loc_idx = self.potential_lava_locs[self.lava_idx[-1]]
            loc = np.squeeze(np.asarray(np.where(loc_idx == _idx)))
            self.put_obj(Lava(), loc[1], loc[0])
        self.lava_idx = np.asarray(self.lava_idx)

        # Set agent pos.
        self.agent_pos = (1, 3)
        self.agent_dir = 0

        self.goal_location = (width - 2, 3)
        self.put_obj(Goal(), *self.goal_location)

        # Get the list of lava locations.
        self.lava_locations = []
        self._lava_idx = []
        for _i in range(len(self.grid.grid)):
            if type(self.grid.grid[_i]) == Lava:
                _idx = np.asarray(np.where(np.arange(width * height).reshape(width, height) == _i)).squeeze()
                self.lava_locations.append((_idx[1], _idx[0]))
                self._lava_idx.append(_i)
        p = 0

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        agent_pos = self.agent_pos
        _idx = np.arange((self.width - 2) * (self.height - 2)).reshape((self.width - 2), (self.height - 2))[
            agent_pos[0] - 1, agent_pos[1] - 1]
        agent_pos_one_hot = np.zeros((self.width - 2) * (self.height - 2))
        agent_pos_one_hot[_idx] = 1.
        compact_state = np.concatenate((agent_pos_one_hot,))

        try:
            goal_pos_one_hot = np.zeros((2,))
            goal_pos_one_hot[self.goal_idx[0]] = 1.
            compact_state = np.concatenate((agent_pos_one_hot, goal_pos_one_hot))
        except:
            pass

        # If we have a gap, append that.
        _lava_one_hot = np.zeros((len(self.potential_lava_locs, )))
        _lava_one_hot[self.lava_idx] = 1
        compact_state = np.concatenate((compact_state, _lava_one_hot))

        return compact_state

    def render(self, mode='human', _key=None):

        _key_to_render = _key if _key is not None else self.render_type

        # Add a partial aerial rendering.
        # Note that partial rendering is also designed such that the agent is always obs-up (as opposed to north up).
        if 'partial_view_rendering' == _key_to_render:
            # This is legacy code and should never be hit.
            raise NotImplementedError

        # Show it a compact rendering (state without the location of the holes)
        elif _key_to_render == 'partial_state':
            agent_pos = self.agent_pos
            _idx = np.arange((self.width - 2) * (self.height - 2)).reshape((self.width - 2), (self.height - 2))[agent_pos[0] - 1, agent_pos[1] - 1]
            agent_pos_one_hot = np.zeros((self.width - 2) * (self.height - 2))
            agent_pos_one_hot[_idx] = 1.

            compact_state = np.concatenate((agent_pos_one_hot,))

            return compact_state

        # Add a partial aerial rendering.
        # Note that partial rendering is also designed such that the agent is always obs-up (as opposed to north up).
        elif 'observe' == _key_to_render:
            try:

                observe = self.special_render(mode='DONTDISPLAY', highlight=False, tile_size=self.tile_size, render_lava=False)  # tilesize sets the fidelity of the rendering.
                observe = observe.astype(np.double) / 255.0   # Need to make obsevrations in the range [0, 1.]
                if self.full_rendering_obs_noise > 0:
                    observe = np.clip(observe + np.random.normal(0, self.full_rendering_obs_noise, observe.shape), a_min=0.0, a_max=1.0)

            except Exception as err:
                observe = None

            return observe

        # Add a full aerial rendering.
        elif 'full_observe' == _key_to_render:
            try:
                full_observe = self.special_render(mode='DONTDISPLAY', highlight=False, tile_size=self.tile_size, render_lava=True)  # tilesize sets the fidelity of the rendering.
                full_observe = full_observe.astype(np.double) / 255.0   # Need to make obsevrations in the range [0, 1.]
                if self.full_rendering_obs_noise > 0:
                    full_observe = np.clip(full_observe + np.random.normal(0, self.full_rendering_obs_noise, full_observe.shape), a_min=0.0, a_max=1.0)
            except:
                full_observe = None

            return full_observe

        else:
            raise NotImplementedError  # Type of rendering not recognised.


class LavaGapS5Env(LavaGapEnv):
    def __init__(self):
        super().__init__(size=5)

class LavaGapS6Env(LavaGapEnv):
    def __init__(self):
        super().__init__(size=6)

class LavaGapS7Env(LavaGapEnv):
    def __init__(self):
        super().__init__(size=7)

class LavaGapS8Env(LavaGapEnv):
    def __init__(self):
        super().__init__(size=8)

class LavaGapS9Env(LavaGapEnv):
    def __init__(self):
        super().__init__(size=9)

register(
    id='MiniGrid-LavaGapS5-v0',
    entry_point='a2d.environments.minigrid.gym_minigrid.envs:LavaGapS5Env'
)

register(
    id='MiniGrid-LavaGapS6-v0',
    entry_point='a2d.environments.minigrid.gym_minigrid.envs:LavaGapS6Env'
)

register(
    id='MiniGrid-LavaGapS7-v0',
    entry_point='a2d.environments.minigrid.gym_minigrid.envs:LavaGapS7Env'
)

register(
    id='MiniGrid-LavaGapS8-v0',
    entry_point='a2d.environments.minigrid.gym_minigrid.envs:LavaGapS8Env'
)

register(
    id='MiniGrid-LavaGapS9-v0',
    entry_point='a2d.environments.minigrid.gym_minigrid.envs:LavaGapS9Env'
)
