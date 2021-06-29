# This code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Licence.  The full licence and information is available at:
# https://github.com/plai-group/a2d/blob/master/LICENCE.md
# © Andrew Warrington, J. Wilder Lavington, Adam Scibior, Mark Schmidt & Frank Wood.
# Accompanies the paper: Robust Asymmetric Learning in POMDPs, ICML 2021.

class EnvManager():
    """
    SOMETHING TO HANDLE DIFFERENT ENVIROMENTS THAT ARE CURRENTLY SUPPORTED,
    ONLY NEEDED WHEN THE ENVIRONMENT BEING HANDLES REQUIRES SOME SORT OF INPUT
    TO BE GENERATED. OTHERWISE THIS JUST SITS AS A UNNECCESARY WRAPPER, WITH
    SOME LOGGING HELPERS.
    """
    def __init__(self, env, wrappers=None):
        # copy all environment attributes
        self.__dict__.update(env.__dict__)
        self.env = env
        # for mujoco
        try:
            self.sim = env.sim
        except:
            pass
        # set wrappers
        if wrappers is None:
            self.wrappers = {}
            self.wrappers['step'] = lambda k: k
            self.wrappers['render'] = lambda k: k
            self.wrappers['reset'] = lambda k: k
        else:
            self.wrappers = wrappers
            assert isinstance(self.wrappers, dict)
            assert all(name in self.wrappers for \
                name in ('step', 'render', 'reset'))

    # core environment functions
    def step(self, action):
        base_output = self.env.step(action)
        return self.wrappers['step'](base_output)

    def render(self, **kwargs):
        base_output = self.env.render(**kwargs)
        return self.wrappers['render'](base_output)

    def reset(self, **kwargs):
        base_output = self.env.reset(**kwargs)
        return self.wrappers['reset'](base_output)
