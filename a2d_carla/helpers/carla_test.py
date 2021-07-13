import contextlib
import os
import sys

import gym
import torch
import signal
import argparse
import logging

from plai_carla import utils, benchmarks, gym_wrapper, scenarios

from plai_rl.algorithms.a2c.agent import ActorCritic
from plai_rl.environments.env_wrap import CarlaEnvWrapper
from plai_rl.algorithms.a2c.envs import make_vec_envs
from plai_rl.algorithms.a2c.agent import Trainer as a2c_trainer

def main():

    # Enable graceful termination
    def sigterm_handler(signum, frame):
        raise InterruptedError("SIGTERM received")

    signal.signal(signal.SIGTERM, sigterm_handler)

    # Define and parse command line arguments
    argparser = argparse.ArgumentParser()
    utils.add_driving_options(argparser)
    gym_wrapper.CarlaEnv.add_command_line_options(argparser)
    argparser.add_argument('--checkpoint_path', type=str, default='a2d_trained')
    subparsers = argparser.add_subparsers(title='available benchmarks', dest='benchmark')
    parser_corl2017 = subparsers.add_parser('corl2017')
    benchmarks.Corl2017Benchmark.add_command_line_options(parser_corl2017)
    parser_nocrash = subparsers.add_parser('nocrash')
    benchmarks.NoCrashBenchmark.add_command_line_options(parser_nocrash)
    parser_occluded = subparsers.add_parser('occluded')
    scenarios.OccludedPedestrianScenario.add_command_line_options(parser_occluded)
    parser_overtaking = subparsers.add_parser('overtaking')
    scenarios.OvertakingTruckScenario.add_command_line_options(parser_overtaking)

    args = argparser.parse_args()

    env_names = dict(
        corl2017='Corl2017',
        nocrash='NoCrash',
        occluded='OccludedPedestrian',
        overtaking='OvertakingTruck',
        waypoint='WaypointFollowing'
    )
    with torch.no_grad():
        params = torch.load(os.path.join(args.checkpoint_path, "args.pt"))
        params.fps = args.fps
        if args.benchmark == 'nocrash':
            params.empty_only = args.empty_only
            params.regular_only = args.regular_only
            params.dense_only = args.dense_only
        params.env_gen = lambda params_: gym.make('plai-carla/%s-v0' % env_names[args.benchmark], args = params)
        envs = make_vec_envs('plai-carla/%s-v0' % env_names[args.benchmark], 100, 1, None, \
                                      "cuda:0", True, params, wrappers=None)
        try:
            env = envs.envs[0]
            actor_critic = ActorCritic(env.observation_space, env.action_space,
                                       base_kwargs={'recurrent': False, 'params': params})
            actor_critic.sampling_dist = 'expert'
            actor_critic.pid_sd = 1
            actor_critic.load_state_dict(torch.load(os.path.join(args.checkpoint_path, "model.pt"),
                                                    map_location="cuda:0"))
            actor_critic.to("cuda:0", dtype=torch.float)

            env.env.render()
            if args.video is not None:
                env.env.render(mode='video', filename=args.video, frames=1000)
            obs = envs.reset()
            print("set")
            for j in range(1000):
                for key in obs.keys():
                    obs[key] = obs[key].float()
                value, action, _, _ = actor_critic.act(obs, deterministic=True, device="cuda:0")
                obs, reward, done, info = envs.step(action)
        finally:
            envs.close()

    sys.exit(0)


if __name__ == "__main__":
    plai_carla_logger = logging.getLogger("plai_carla")
    plai_carla_logger.setLevel(logging.INFO)
    main()
