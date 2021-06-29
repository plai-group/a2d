# This file was originally released by Chevalier-Boisvert, Maxime and Willems, Lucas and Pal, Suman
# as part of the gym-minigrid environment, released under the Apache 2.0 Licence.  We have
# modified this file, and distribute this modified file as permitted under the terms of the original
# licence.  The modifications are therefore also covered under the CC licence we release under.

# When referencing our work, please also direct citations/references towards the original authors
# of the environment.  Information is available here: https://github.com/maximecb/gym-minigrid.


from gym.envs.registration import register as gym_register

env_list = []

def register(
    id,
    entry_point,
    reward_threshold=0.95
):
    assert id.startswith("MiniGrid-")
    assert id not in env_list

    # Register the environment with OpenAI gym
    gym_register(
        id=id,
        entry_point=entry_point,
        reward_threshold=reward_threshold
    )

    # Add the environment to the set
    env_list.append(id)
