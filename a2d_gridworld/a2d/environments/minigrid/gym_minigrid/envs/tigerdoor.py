# This file is based heavily on one released by Chevalier-Boisvert, Maxime and Willems, Lucas and Pal, Suman
# as part of the gym-minigrid environment, released under the Apache 2.0 Licence.  We have
# modified this file, and distribute this modified file as permitted under the terms of the original
# licence.  The modifications are therefore also covered under the CC licence we release under.

# When referencing our work, please also direct citations/references towards the original authors
# of the environment.  Information is available here: https://github.com/maximecb/gym-minigrid.


from a2d.environments.minigrid.gym_minigrid.minigrid import *
from a2d.environments.minigrid.gym_minigrid.register import register


class TigerDoorEnv(MiniGridEnv):
    """
    Environment with one wall of lava with a small gap to cross through
    This environment is similar to LavaCrossing but simpler in structure.
    """

    def __init__(self, size=7, obstacle_type=Lava, seed=None):
        self.obstacle_type = obstacle_type

        # Set the maximum runlength of the env.
        _tmax = 100

        super().__init__(
            grid_size=size,
            max_steps=_tmax,
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

        # Set the renderer up.
        self.render_type = 'observe'
        self.tile_size = 6

        # Start with the lights off.
        self.illuminated = False

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

        compact_state = np.concatenate((compact_state, self.illuminated * np.ones(1, )))  # Append lights are off.
        return compact_state

    def _gen_grid(self, width, height, _seed=None):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place barriers.
        _idx = np.arange(width * height).reshape(width, height).T
        _wall_locations = [    15, 22, 29, 36,
                               16,
                                       31, 38,
                               18,
                           12, 19, 26, 33, 40]
        for _l in _wall_locations:
            loc = np.squeeze(np.asarray(np.where(_l == _idx)))
            self.put_obj(Wall(), loc[1], loc[0])

        self.potential_lava_locs = ((5, 2), (5, 4))
        self.potential_goal_locs = ((5, 2), (5, 4))

        # Set which door the tiger (lava) is behind...
        if np.random.rand() < 0.5:
            self.lava_idx = [1]
            self.goal_idx = [0]
            self.lava_location = (5, 2)
            self.goal_location = (5, 4)
        else:
            self.lava_idx = [0]
            self.goal_idx = [1]
            self.lava_location = (5, 4)
            self.goal_location = (5, 2)

        self.goal = Goal()
        self.lava = Lava()
        self.butt = Button()

        self.put_obj(self.lava, *self.lava_location)
        self.put_obj(self.goal, *self.goal_location)
        self.put_obj(self.butt, *(1, 4))

        self.lava_idx = np.asarray(self.lava_idx)
        self.goal_idx = np.asarray(self.goal_idx)

        self.lava_locations = np.asarray(self.lava_idx)

        # Start the agent in a random direction.
        self.agent_pos = (1, 1)  # height // 2)  # np.array((width - 2, height - 2))
        self.agent_dir = 0  # self._rand_int(0, 4)  # This doesn't really do much, though

        # Start with the lights off.
        self.illuminated = False

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

    def render(self, mode='human', _key=None):

        _key_to_render = _key if _key is not None else self.render_type

        # Show it a compact rendering (state without the location of the holes)
        if _key_to_render == 'partial_state':
            agent_pos = self.agent_pos
            _idx = np.arange((self.width - 2) * (self.height - 2)).reshape((self.width - 2), (self.height - 2))[agent_pos[0] - 1, agent_pos[1] - 1]
            agent_pos_one_hot = np.zeros((self.width - 2) * (self.height - 2))
            agent_pos_one_hot[_idx] = 1.
            compact_state = np.concatenate((agent_pos_one_hot,))

            if not self.illuminated:
                compact_state = np.concatenate((compact_state, np.zeros(2, )))  # Append goal.
                compact_state = np.concatenate((compact_state, np.zeros(2, )))  # Append lava.
                compact_state = np.concatenate((compact_state, np.zeros(1, )))  # Append lights are off.
            else:
                _goal_one_hot = np.zeros((len(self.potential_goal_locs, )))
                _goal_one_hot[self.goal_idx] = 1  # np.round((_l[0] * self.width) + _l[1]).astype(np.int)] = 1.
                compact_state = np.concatenate((compact_state, _goal_one_hot))

                _lava_one_hot = np.zeros((len(self.potential_lava_locs, )))
                _lava_one_hot[self.lava_idx] = 1  # np.round((_l[0] * self.width) + _l[1]).astype(np.int)] = 1.
                compact_state = np.concatenate((compact_state, _lava_one_hot))

                compact_state = np.concatenate((compact_state, np.ones(1, )))  # Append lights are on.

            return compact_state

        # Add a partial aerial rendering.
        # Note that partial rendering is also designed such that the agent is always obs-up (as opposed to north up).
        elif 'observe' == _key_to_render:

            try:
                # tilesize sets the fidelity of the rendering.
                observe = self.special_render(mode='DONTDISPLAY', highlight=False, tile_size=self.tile_size, render_lava=self.illuminated, render_goal=self.illuminated)

                observe = observe.astype(np.double) / 255.0   # Need to make obsevrations in the range [0, 1.]
                if self.full_rendering_obs_noise > 0:
                    observe = np.clip(observe + np.random.normal(0, self.full_rendering_obs_noise, observe.shape), a_min=0.0, a_max=1.0)

            except Exception as err:
                observe = None

            return observe

        # Add a full aerial rendering.
        elif 'full_observe' == _key_to_render:
            try:
                # tilesize sets the fidelity of the rendering.
                full_observe = self.special_render(mode='DONTDISPLAY', highlight=False, tile_size=self.tile_size, render_lava=True, render_goal=True)
                full_observe = full_observe.astype(np.double) / 255.0   # Need to make obsevrations in the range [0, 1.]
                if self.full_rendering_obs_noise > 0:
                    full_observe = np.clip(full_observe + np.random.normal(0, self.full_rendering_obs_noise, full_observe.shape), a_min=0.0, a_max=1.0)
            except Exception as err:
                print('Error in render: ', err)
                full_observe = None

            return full_observe

        elif 'state' == _key_to_render:
            return self.gen_obs()

        else:
            raise NotImplementedError  # Type of rendering not recognised.


register(
    id='MiniGrid-TigerDoorEnv-v0',
    entry_point='a2d.environments.minigrid.gym_minigrid.envs:TigerDoorEnv'
)
