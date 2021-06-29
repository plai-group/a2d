# This file was originally released by Chevalier-Boisvert, Maxime and Willems, Lucas and Pal, Suman
# as part of the gym-minigrid environment, released under the Apache 2.0 Licence.  We have
# modified this file, and distribute this modified file as permitted under the terms of the original
# licence.  The modifications are therefore also covered under the CC licence we release under.

# When referencing our work, please also direct citations/references towards the original authors
# of the environment.  Information is available here: https://github.com/maximecb/gym-minigrid.


from a2d.environments.minigrid.gym_minigrid.minigrid import *
from a2d.environments.minigrid.gym_minigrid.register import register

# Set a function for seeding the env.
SEED_FUNC = lambda: int(time.time() * 100000 * os.getpid()) % 2 ** 32 - 1


class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height, _seed=None):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)


        # NOTE - changed the placing of the agent.
        # Place a goal square in the bottom-right corner.
        _rng_state = self.get_np_state()
        if _seed is None:
            try:
                self.seed(int(time.time() * 100000 * os.getpid()) % 2**32 - 1)
            except:
                raise RuntimeError  # Not quite sure why this got hit.
        else:
            self.seed(_seed)

        _goal_location = np.asarray((self.np_random.randint(1, width - 1),
                                     self.np_random.randint(1, height - 1)))

        # _goal_location = np.asarray((width - 2, height - 2))

        self.set_np_state(_rng_state)

        self.goal_location = _goal_location
        # / NOTE - changed the placing of the agent.

        self.put_obj(Goal(), self.goal_location[0], self.goal_location[1])

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            if hasattr(self, 'SEED_FUNC'):
                self.place_agent(_seed=self.SEED_FUNC())
            else:
                self.place_agent(_seed=SEED_FUNC())

        self.mission = "get to the green goal square"

    def get_state_representation(self):
        """
        AW - lifted from __str__
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # # Map of object types to short string
        # OBJECT_TO_STR = {
        #     'wall'          : 5,
        #     'door'          : 6,
        #     'goal'          : 7,
        # }

        # Map of object types to short string
        OBJECT_TO_STR = {
            # 'wall'          : 5,
            # 'door'          : 6,
            'goal'          : 1,
        }

        # Short string for opened door
        OPENDED_DOOR_IDS = '_'

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {
            0: 1,
            1: 2,
            2: 3,
            3: 4
        }

        state = []

        for j in range(1, self.grid.height-1):

            for i in range(1, self.grid.width-1):

                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    if AGENT_DIR_TO_STR[self.agent_dir] is not None:
                        state_vec = [0] * len(AGENT_DIR_TO_STR)
                        state_vec[AGENT_DIR_TO_STR[self.agent_dir]-1] = 1
                        state.extend(state_vec)
                    else:
                        raise RuntimeError  # What is this for??
                else:
                    state.extend([0] * len(AGENT_DIR_TO_STR))

                # Get the current state.
                c = self.grid.get(i, j)

                # c being none indicates is it just floor.
                if c is None:
                    state.append(0)
                    continue

                # if c.type == 'door':
                #     if c.is_open:
                #         str += '__'
                #     elif c.is_locked:
                #         str += 'L' + c.color[0].upper()
                #     else:
                #         str += 'D' + c.color[0].upper()
                #     continue

                if OBJECT_TO_STR[c.type] is not None:
                    state.append(OBJECT_TO_STR[c.type])

        return np.asarray(state)

class EmptyEnv5x5(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=5, **kwargs)

class EmptyRandomEnv5x5(EmptyEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)

class EmptyEnv6x6(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=6, **kwargs)

class EmptyRandomEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)

class EmptyRandomEnv8x8(EmptyEnv):
    def __init__(self):
        super().__init__(size=8, agent_start_pos=None)

class EmptyEnv16x16(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, **kwargs)

class EmptyRandomEnv16x16(EmptyEnv):
    def __init__(self):
        super().__init__(size=16, agent_start_pos=None)

register(
    id='MiniGrid-Empty-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyEnv5x5'
)

register(
    id='MiniGrid-Empty-Random-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnv5x5'
)

register(
    id='MiniGrid-Empty-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyEnv6x6'
)

register(
    id='MiniGrid-Empty-Random-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnv6x6'
)

register(
    id='MiniGrid-Empty-8x8-v0',
    entry_point='gym_minigrid.envs:EmptyEnv'
)

register(
    id='MiniGrid-Empty-Random-8x8-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnv8x8'
)

register(
    id='MiniGrid-Empty-16x16-v0',
    entry_point='gym_minigrid.envs:EmptyEnv16x16'
)

register(
    id='MiniGrid-Empty-Random-16x16-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnv16x16'
)
