# This code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Licence.  The full licence and information is available at:
# https://github.com/plai-group/a2d/blob/master/LICENCE.md
# © Andrew Warrington, J. Wilder Lavington, Adam Scibior, Mark Schmidt & Frank Wood.
# Accompanies the paper: Robust Asymmetric Learning in POMDPs, ICML 2021.

# general imports
import timeit
import shutil

from tests.A2D.A2Dagger_arguments import get_args as get_a2d_args
from tests.A2D.util import make_logging, do_rl, do_a2d, do_advanced_rl

import matplotlib.pyplot as plt
from platform import system
if ('Linux' in system()) or ('linux' in system()):
    plt.switch_backend('agg')


def main():
    """
    
    :return: 
    """

    """ Configure arguments for all experiments. ------------------------------------------------------------------- """

    # Get the args, but force it to use observe implementation by default.
    default_args = get_a2d_args()

    # Make the logging folders.
    default_args, tee = make_logging(default_args)

    """ Configure expert ------------------------------------------------------------------------------------------- """

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

        if expert_args.rl_method == 'trpo':

            # # This is the old RL implementation that shipped as part of the library originally.
            # run_trpo(expert_args)

            do_rl(args=expert_args,
                  _render_type='state',
                  _return_type='state',
                  _frame_stack=1)

        else:
            raise NotImplementedError
    else:
        expert_source = 'tests/results/AdaptAsymDagger/' + default_args.env_name
        print('\n\n\n>> Copy expert from: {}'.format(expert_source + '/expert.pt'))
        print('>> Copy expert to  : {}'.format(default_args.expert_location + '/expert.pt'))
        shutil.copy(expert_source + '/expert.pt', default_args.expert_location + '/expert.pt')

    """ Debugging experiments on full state. ----------------------------------------------------------------------- """

    # Test straightforward symmetric DAgger.
    if default_args.d_state:
        tee.tag = '02_DAGGER_STATE'
        print('\n\n\n>> DAgger: Test straightforward symmetric DAgger.')
        do_a2d(args=default_args,
               _mode='D',
               _render_type='state',
               _frame_stack=1,
               _cotrain=False)

    # Run A2DAgger on state. Little bit pointless, but lets just double check. 
    if default_args.a2d_state:
        tee.tag = '03_A2D_STATE'
        print('\n\n\n>> A2D: Run A2DAgger on state. Little bit pointless, but lets just double check.')
        do_a2d(args=default_args,
               _mode='A2D',
               _render_type='state',
               _frame_stack=1,
               _cotrain=default_args.cotrain)


    """ Now run on partial state ------------------------------------------------------------------------------------ """

    # Run RL on the partial state. 
    if default_args.rl_partial_state:
        tee.tag = '04_RL_PART_STATE'
        print('\n\n\n>> RL: Run RL on the partial state.')
        do_rl(args=default_args,
              _render_type='partial_state', 
              _return_type='partial_state', 
              _frame_stack=default_args.frame_stack)

    # Run Asymmetric DAgger on partial state. 
    if default_args.ad_partial_state:
        tee.tag = '05_AD_PART_STATE'
        print('\n\n\n>> AD: Run AD on partial state.')
        do_a2d(args=default_args,
               _mode='AD', 
               _render_type='partial_state', 
               _frame_stack=default_args.frame_stack, 
               _cotrain=False)

    # Run A2DAgger on partial state. 
    if default_args.a2d_partial_state:
        tee.tag = '06_A2D_PART_STATE'
        print('\n\n\n>> A2D: Run A2DAgger on partial state.')
        do_a2d(args=default_args,
               _mode='A2D', 
               _render_type='partial_state', 
               _frame_stack=default_args.frame_stack, 
               _cotrain=default_args.cotrain)


    """ Now run on partial observe -- these are more expensive------------------------------------------------------ """

    # Run Asymmetric DAgger on partial observe. 
    if default_args.ad_partial_observe:
        tee.tag = '07_AD_PART_OBSERVE'
        print('\n\n\n>> AD: Run Asymmetric DAgger on partial observe. Expensive experiment...')
        do_a2d(args=default_args,
               _mode='AD', 
               _render_type='observe', 
               _frame_stack=default_args.frame_stack, 
               _cotrain=False)

    # Run A2DAgger on partial observe. 
    if default_args.a2d_partial_observe:
        tee.tag = '08_A2D_PART_OBSERVE'
        print('\n\n\n>> A2D: Run A2DAgger on partial observe. Expensive experiment...')
        do_a2d(args=default_args,
               _mode='A2D', 
               _render_type='observe', 
               _frame_stack=default_args.frame_stack, 
               _cotrain=default_args.cotrain)

    # Run RL with a pre-trained encoder. 
    if default_args.ete_partial_observe:
        tee.tag = '09_PRE_ENC_PART_OBSERVE'
        print('\n\n\n>> RL: Run RL with pretrained encoder on observe. Expensive experiment...')
        do_advanced_rl(args=default_args,
                       _render_type='observe',
                       _frame_stack=default_args.frame_stack,
                       _return_type='observe',
                       _ete=True)

    # Run ARL. 
    if default_args.arl_partial_observe:
        tee.tag = '10_ARL_PART_OBSERVE'
        print('\n\n\n>> RL: Run ARL on partial observe. Expensive experiment...')
        do_advanced_rl(args=default_args,
                       _render_type='observe', 
                       _frame_stack=default_args.frame_stack, 
                       _return_type='observe', 
                       _ete=False)

    # Run RL on the partial observe. 
    if default_args.rl_partial_observe:
        tee.tag = '11_RL_PART_OBSERVE'
        print('\n\n\n>> RL: Run RL on the partial observe. Expensive experiment...')
        do_rl(args=default_args,
              _render_type='observe', 
              _return_type='observe', 
              _frame_stack=default_args.frame_stack)


    """ Now run on full observe -- these are MUCH more expensive, so do these last --------------------------------- """

    # Run Asymmetric DAgger on partial observe. 
    if default_args.ad_full_observe:
        tee.tag = '12_AD_FULL_OBSERVE'
        print('\n\n\n>> AD: Run Asymmetric DAgger on full observe. Expensive experiment...')
        do_a2d(args=default_args,
               _mode='AD', 
               _render_type='full_observe', 
               _frame_stack=default_args.frame_stack, 
               _cotrain=False)

    # Run A2DAgger on partial observe. 
    if default_args.a2d_full_observe:
        tee.tag = '13_A2D_FULL_OBSERVE'
        print('\n\n\n>> A2D: Run A2DAgger on full observe. Expensive experiment...')
        do_a2d(args=default_args,
               _mode='A2D', 
               _render_type='full_observe', 
               _frame_stack=default_args.frame_stack, 
               _cotrain=default_args.cotrain)

    # Run RL on the partial observe. 
    if default_args.rl_full_observe:
        tee.tag = '14_RL_FULL_OBSERVE'
        print('\n\n\n>> RL: Run RL on the full observe. WARNING: VERY EXPENSIVE EXPERIMENT...')
        do_rl(args=default_args,
              _render_type='observe', 
              _return_type='full_observe', 
              _frame_stack=default_args.frame_stack)

    # Run A2DAgger on partial observe.
    if default_args.arl_full_observe:
        tee.tag = '15_ARL_FULL_OBSERVE'
        print('\n\n\n>> ARL: Run ARL on full observe. Slightly pointless, expensive experiment...')
        do_advanced_rl(args=default_args,
                       _render_type='observe',
                       _return_type='full_observe',
                       _frame_stack=default_args.frame_stack,
                       _ete=False)

    # Run RL on the partial observe.
    if default_args.ete_full_observe:
        tee.tag = '16_PRE_ENC_FULL_OBS'
        print('\n\n\n>> PreEnc: Run RL on the full observe. Slightly pointless, expensive experiment...  '
              'Lets one confirm to oneself that the encoder architecture and everything is working, though.')
        do_advanced_rl(args=default_args,
                       _render_type='full_observe',
                       _return_type='full_observe',
                       _frame_stack=default_args.frame_stack,
                       _ete=True)


    """ Do any cleanup. -------------------------------------------------------------------------------------------- """

    # Print final time.
    tee.tag = 'Finishing...'
    print('\n\n\n>> Training complete, time elapsed: {}'.format(timeit.default_timer() - default_args._st))


if __name__ == "__main__":
    main()
