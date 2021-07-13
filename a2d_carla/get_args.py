
# general imports
import argparse
import torch
import os

# main arg-parse
def get_args():

    #
    parser = argparse.ArgumentParser(description='A2D Parser')

    # pick what we are running
    parser.add_argument('--generate_video', type=int, default=0, help='')
    parser.add_argument('--skip_training', type=int, default=0, help='')
    parser.add_argument('--actor_critic_path', type=str, default='./trained_models/scenario_1/...', help='')
    parser.add_argument('--video_path', type=str, default='./trained_models/scenario_1/...', help='')
    parser.add_argument('--video_file_name', type=str, default='video_ex.mp4', help='')
    parser.add_argument('--save_intermediate_video', type=int, default=0, help='')

    # experiment arguments
    parser.add_argument('--num_env_steps', type=int, default=1e6, help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--num_processes', type=int, default=5, help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num_steps', type=int, default=50, help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--max_time_horizon', type=int, default=2000, help='')
    parser.add_argument('--eval_model', type=int, default=1, help='')

    # coefficients
    parser.add_argument('--entropy_coeff', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value_loss_coeff', type=float, default=1., help='value loss coefficient (default: 0.5)')
    parser.add_argument('--encoder_coeff', type=float, default=1., help='')
    parser.add_argument('--action_loss_coeff', type=float, default=1., help='')

    # agent model
    parser.add_argument('--agent_frame_stack', type=int, default=5, metavar='G', help='')
    parser.add_argument('--agent_convlayers', type=int, default=5, help='')
    parser.add_argument('--agent_stride', type=int, default=2, help='')
    parser.add_argument('--agent_mlplayers', type=int, default=1, help='')
    parser.add_argument('--agent_numfilters', type=int, default=32, help='')
    parser.add_argument('--agent_kernelsize', type=int, default=3, help='')
    parser.add_argument('--agent_featuredim', type=int, default=250, help='')
    parser.add_argument('--agent_hidden_size', type=int, default=256, help='')

    # expert model
    parser.add_argument('--expert_frame_stack', type=int, default=5, help='')
    parser.add_argument('--expert_convlayers', type=int, default=5, help='')
    parser.add_argument('--expert_stride', type=int, default=2, help='')
    parser.add_argument('--expert_mlplayers', type=int, default=1, help='')
    parser.add_argument('--expert_numfilters', type=int, default=32, help='')
    parser.add_argument('--expert_kernelsize', type=int, default=3, help='')
    parser.add_argument('--expert_featuredim', type=int, default=250, help='')
    parser.add_argument('--expert_hidden_size', type=int, default=1024, help='')

    # advanatge estimation
    parser.add_argument('--use_gae', type=int, default=1, help='use generalized advantage estimation')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--use_proper_time_limits', type=int, default=0, help='compute returns taking into account time limits')
    parser.add_argument('--ppo_clip', type=float, default=0.2, help='compute returns taking into account time limits')

    # optim hyper-parameters
    parser.add_argument('--single_optim', type=int, default=0, help='')
    parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')

    # learning rate stuff
    parser.add_argument('--use_linear_lr_decay', type=int, default=0, help='use a linear schedule on the learning rate')
    parser.add_argument('--sd_lr_scaling', type=float, default=0.5, help='max number of steps')
    parser.add_argument('--log_lr', type=float, default=-4, help='learning rate (default: 7e-4)')

    # conservative policy iteration to ensure critic correctness
    parser.add_argument('--pretrain_critic_updates', type=int, default=0, help='')
    parser.add_argument('--delayed_policy_update', type=int, default=-1, help='')
    parser.add_argument('--critic_updates', type=int, default=5, help='')
    parser.add_argument('--policy_updates', type=int, default=5, help='')
    parser.add_argument('--critic_batch_size', type=int, default=64, help='')
    parser.add_argument('--policy_batch_size', type=int, default=64, help='')
    parser.add_argument('--beta_update_interval', type=int, default=25, help='')
    parser.add_argument('--RL_updates_per_batch', type=int, default=1, help='')

    # Online Imitation Learning
    parser.add_argument('--AD_updates_per_batch', type=int, default=5, help='')
    parser.add_argument('--AD_batch_size', type=int, default=512, help='')
    parser.add_argument('--AD_cotrain_vf', type=int, default=0, help='')
    parser.add_argument('--AD_cotrain_qf', type=int, default=0, help='')
    parser.add_argument('--AD_buffer_mem', type=int, default=10000, help='')
    parser.add_argument('--AD_full_mem', type=int, default=0, help='')
    parser.add_argument('--use_kl_dagger', type=int, default=1, help='update vf and actor simultaneously')

    # filter / clip states and rewards
    parser.add_argument('--norm_states', type=float, default=0, metavar='G', help='')
    parser.add_argument('--norm_rewards', type=str, default='rewards', metavar='G', help='')
    parser.add_argument('--clip_obs', type=float, default=-1, help='clip observations')
    parser.add_argument('--clip_rew', type=float, default=-1, help='clip rewards')
    parser.add_argument('--filter_state', type=int, default=0, metavar='P', help='')
    parser.add_argument('--filter_rewards', type=int, default=0, metavar='P', help='')

    # AD stuff
    parser.add_argument('--fixed_expert_sd', type=float, default=0.005, help='')
    parser.add_argument('--beta', type=float, default=1., help='')
    parser.add_argument('--beta_update', type=float, default=0.9999, help='')

    # wnadb stuff
    parser.add_argument('--sweep_id', type=int, default=0, help='load a pre-trained model (default: 0)')
    parser.add_argument('--id', type=str, default='test-0.01', help='Enforce a resume on wandb.')
    parser.add_argument('--project', type=str, default='random_tests', help='')
    parser.add_argument('--group', type=str, default='random_tests', help='')

    # logging
    parser.add_argument('--logged_moving_avg', type=int, default=25, metavar='G', help='')
    parser.add_argument('--log_interval', type=int, default=100, help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--eval_interval', type=int, default=10000, help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--num_evals', type=int, default=10, help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save_interval', type=int, default=1000, help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--log_dir', default='./results/', help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save_dir', default='./trained_models/', help='directory to save agent logs (default: ./trained_models/)')

    # other crap
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--cuda_deterministic', type=int, default=0, help="sets flags for determinism when using CUDA (potentially slow!)")

    ## for carla
    parser.add_argument('--straight_only', action='store_true', help='Run only lane following scenarios')
    parser.add_argument('--turn_only',action='store_true',help='Run only single turn scenarios')
    parser.add_argument('--navigate_only',action='store_true', help='Run only navigation scenarios')
    parser.add_argument('--dynamic_only',action='store_true', help='Run only dynamic scenarios')
    parser.add_argument('--limit_scenarios_to',type=int,default=None,help='How many scenarios to run (of each kind). Run all by default.')
    parser.add_argument('--force_fps', type=float, default=None, help='Starting position of the oncoming vehicle')
    parser.add_argument('--use_compressed_state', type=int, default=1, help='max number of steps')
    parser.add_argument('--aux_info_stack', type=int, default=25, metavar='G', help='')

    # scenario specific args arg
    parser.add_argument('--pedestrian_prob', type=float, default=0.5, help='Probability of the pedestrian being present')
    parser.add_argument('--pedestrian_speed', type=float, default=None, help='Speed of pedestrian (when present)')
    parser.add_argument('--opposite_position', type=float, default=None, help='Starting position of the oncoming vehicle')
    parser.add_argument('--tracks_folder', type=str, default='/opt/carla-dataset/Town01')

    # end criteria
    parser.add_argument('--end_on_invasion', type=int, default=1, help='')
    parser.add_argument('--end_on_collision', type=int, default=1, help='')

    # reward info
    parser.add_argument('--expert_reward', type=int, default=0, help='')
    parser.add_argument('--waypoint_reward', type=int, default=0, help='')
    parser.add_argument('--completed_reward', type=int, default=1, help='')
    parser.add_argument('--survive_reward', type=int, default=0, help='')
    parser.add_argument('--nominal_penalty', type=int, default=0, help='')
    parser.add_argument('--clipped_nominal_reward', type=int, default=0, help='')
    parser.add_argument('--action_diff_penalty', type=int, default=0, help='')

    # penalties
    parser.add_argument('--collision_penalty', type=int, default=0, help='')
    parser.add_argument('--invasion_penalty', type=int, default=0, help='')
    parser.add_argument('--action_penalty', type=int, default=1, help='')
    # parse
    args, _ = parser.parse_known_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    args.num_gpu = torch.cuda.device_count()
    args.rotate_birdview = True

    # return it all
    return args, parser
