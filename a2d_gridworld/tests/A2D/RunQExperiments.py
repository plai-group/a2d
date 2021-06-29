# This code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Licence.  The full licence and information is available at:
# https://github.com/plai-group/a2d/blob/master/LICENCE.md
# © Andrew Warrington, J. Wilder Lavington, Adam Scibior, Mark Schmidt & Frank Wood.
# Accompanies the paper: Robust Asymmetric Learning in POMDPs, ICML 2021.

# general imports
import timeit
import shutil

from tests.A2D.Q_arguments import get_args as get_q_args
from tests.A2D.util import make_logging, do_rl, do_a2d

import matplotlib.pyplot as plt
from platform import system
if ('Linux' in system()) or ('linux' in system()):
    plt.switch_backend('agg')


def main():
    """
    This is a stripped back version of RunA2DExperiments.py for reproduction of the
    Q experiments included in the supplement.  The functionality is basically the same,
    it just wraps a limited range of different functions, and, calls the right default
    arguments dictionary by default.
    :return: 
    """

    # Get the args.
    default_args = get_q_args()

    # Make the logging folders.
    default_args, tee = make_logging(default_args)

    # Generate a new expert so experiments are fully modular.
    tee.tag = '00_EXPERT_PREP'
    expert_args, env = do_rl(default_args,
                             _render_type='state',
                             _return_type='state',
                             _frame_stack=1,
                             _run_rl=False)

    default_args.expert_location = expert_args.log_dir
    expert_args.expert_location = default_args.expert_location

    if default_args.rl_state:
        print('\n\n\n>> RL: Train new expert.')
        tee.tag = '01_RL_STATE'

        expert_args.save_as_expert = True

        do_rl(args=expert_args,
              _render_type='state',
              _return_type='state',
              _frame_stack=1)

    else:
        expert_source = 'tests/results/AdaptAsymDagger/' + default_args.env_name
        print('\n\n\n>> Copy expert from: {}'.format(expert_source + '/expert.pt'))
        print('>> Copy expert to  : {}'.format(default_args.expert_location + '/expert.pt'))
        shutil.copy(expert_source + '/expert.pt', default_args.expert_location + '/expert.pt')

    # Run RL on the partial state. 
    if default_args.rl_partial_state:
        tee.tag = '04_RL_PART_STATE'
        print('\n\n\n>> RL: Run RL on the partial state.')
        do_rl(args=default_args,
              _render_type='partial_state', 
              _return_type='partial_state', 
              _frame_stack=default_args.frame_stack)

    # Run A2DAgger on partial state. 
    if default_args.a2d_partial_state:
        tee.tag = '06_A2D_PART_STATE'
        print('\n\n\n>> A2D: Run A2DAgger on partial state.')
        do_a2d(args=default_args,
               _mode='A2D', 
               _render_type='partial_state', 
               _frame_stack=default_args.frame_stack, 
               _cotrain=default_args.cotrain)

    # Print final time.
    tee.tag = 'Finishing...'
    print('\n\n\n>> Training complete, time elapsed: {}'.format(timeit.default_timer() - default_args._st))


if __name__ == "__main__":
    main()
