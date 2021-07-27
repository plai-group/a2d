# AdaptiveAsymmetricDAgger
A2D implementation in CARLA with two unique scenarios. Each of these scenarios tests a different aspect of asymmetric behavior. In the first scenario, we test the algorithms ability to avoid the type of sub-optimal behavior found in ice-lake (e.g. avoiding areas of high uncertainty). In the second we present a much more difficult instantiation of the tiger door problem where the agent must actively seek out to reduce the uncertainty within its environment. In this case by peaking out from behind a truck to see if it is safe to pass a truck before doing so.

In the first scenario named OccludedPedestrian-v0, the agent is directed to exactly follow the behavior of a PID controller as it passes a truck which with some non-zero probability has a pedestrian behind it. In this setting, the feedback for the agent is given by the front-view camera alone, while the expert is provided with a compact representation of state. The reward by this environment is given by the L2 loss between the policy actions, and the PID controller, as well as for reaching a new waypoint or completing the task completely. Stopping conditions for this environment include a time-out for when the agent does not pass enough waypoints quickly enough, a maximum time-limit, as well as a collision, and lane invasion.

In the first scenario named OvertakingTruck-v0, the agent is learned through a combination of shaped rewards, and strict stopping conditions. In this scenario, if the agent gets to far behind the truck which it is supposed to follow, the episode will terminate. Additionally, if there are any collisions wither with the guard rail or other vehicles the episode will end. Reward in this environment is provided for survival (e.g. not crashing and not falling behind), as well as the difference in road position between the truck that must be passed and the ego vehicle. This ensures that the agent is incentivized to learn how to safely pass the truck.

## Repo Overview
This repository can be broken up into a couple crucial components. The first is the library itself, which will be discussed in more detail below. the second is the set of helper files, the first of which `helpers/sweep_runner.py` is used for a SLURM based cluster, and allows for directly submitting jobs which will execute a specific command. Examples of such commands are given below, but any `python3` or `wandb sweep ...` will suffice. This folder also includes a program `helpers/carla_step.py` which will generate a video if called. The `plotting` folder stores all videos, plots and data generated by experimentation. Additionally, most trained models will be stored in `trained_models`. Lastly, to run the majority of scripts you only need to call `main.py`, which will itself call `get_args.py`. `get_args.py` allows the user to ignore many of the standard parameters used in many of the algorithms found here.

## Library Installation
make sure to download the required singularity container before running any of the experiments below. With this singularity container downloaded, and included inside `../$HOME_DIR/$A2D_LIB_PATH/AdaptiveAsymmetricDAgger` to install the library something like the following:
```
cd ../$HOME_DIR/$A2D_LIB_PATH/AdaptiveAsymmetricDAgger
pip install -e .
```

## Library Overview
* carla_a2d
>* algorithms (algorithm instantiations for PPO, A2D, AD, ect)
>>* a2d_class
>>* ad_class
>>* rl_class
>* baselines (baselines library instance for multiproccesesing)
>* ..... see baselines docs ....
>* environments (environment wrappers and associated helpers)
>>* env_wrap
>>* envs
>>* env_utils
>* evaluation (functions to vizualize and evaluate)
>>* evaluate_agents
>>* visualization
>* memory (trajectory information storage and custom data loaders)
>>* storage
>* models (nueral network models and pytorch distributions)
>>* distributions
>>* model
>* utils (helper functions)
>>* gen_utils
>>* optim_utils
>>* torch_utils
>* agent (function which manages different training algorithms and initialization)
>* evaluator (function which manages different evaluation and vizualization programs)
>* trainer (function which wraps agent and evalutor to create interpretable test scripts)

## Scenario 1: OccludedPedestrian

In order to generate scripts which will automatically generate the desired parameter set, run the include the following commands. Each of which will create a unique sweep ID, that can be used to run the first scenario with a specific algorithm.
```
wandb sweep sweeps/scenario_1/scenario_1_rl_agent.yaml # RL - Agent
wandb sweep sweeps/scenario_1/scenario_1_rl_expert.yaml # RL - Expert
wandb sweep sweeps/scenario_1/scenario_1_ad_agent.yaml # AD - Agent
wandb sweep sweeps/scenario_1/scenario_1_ad_expert.yaml # AD - Expert
wandb sweep sweeps/scenario_1/scenario_1_a2d_agent.yaml # A2D
```
Running any one of these commands will generate something similar to the following:
```
wandb: Creating sweep from: sweeps/scenario_1/scenario_1_<algo-key>.yaml
wandb: Created sweep with ID: <unique-id>
wandb: View sweep at: https://wandb.ai/<wandb-account>/scenario_1_a2d_ICML/sweeps/<unique-id>
wandb: Run sweep agent with: wandb agent /<wandb-account>/scenario_1_a2d_ICML/<unique-id>
```
In order to run this set of parameters, and log the evaluation out to weights and biases, you need to use the singularity container associated with this set of runs. To use this container, make sure that the "--nv" and "-B" are included or the container will likely not run correctly.
```
singularity run --nv -B $(pwd) a2d.sif wandb agent --count 1 <wandb-account>/scenario_1_a2d_ICML/<unique-id>
```
Alternatively, programs can be run more manually by directly calling on the main function:
```
singularity run --nv -B $(pwd) a2d.sif python main.py --AD_batch_size=64 --AD_buffer_mem=25000 --AD_full_mem=1 --AD_updates_per_batch=10 --action_penalty=1 --agent_frame_stack=1 --algo_key=a2d-agent --aux_info_stack=1 --beta=0 --clipped_nominal_reward=0 --collision_penalty=1 --completed_reward=1 --critic_batch_size=64 --critic_updates=3 --delayed_policy_update=1 --entropy_coeff=0.003 --env_name=plai-carla/OccludedPedestrian-v0 --eps=8.5e-05 --eval_interval=4000 --expert_frame_stack=1 --expert_reward=1 --force_fps=10 --gae_lambda=0.91 --gamma=0.98 --group=a2d_agent_eval --invasion_penalty=1 --log_dir=./trained_models/scenario_1/a2d-agent-eval/ --log_interval=1 --log_lr=-4 --logged_moving_avg=25 --max_grad_norm=1.25 --max_time_horizon=90 --nominal_penalty=0 --norm_states=1 --num_env_steps=400000 --num_processes=4 --num_steps=512 --policy_batch_size=64 --policy_updates=5 --pretrain_critic_updates=1 --save_dir=./trained_models/scenario_1/a2d-agent-eval/ --save_intermediate_video=0 --save_interval=10 --seed=1 --survive_reward=0 --use_compressed_state=1 --use_gae=1 --use_linear_lr_decay=0 --value_loss_coeff=1.25 --waypoint_reward=1
```


## Scenario 2: OvertakingTruck

In order to generate scripts which will automatically generate the desired parameter set, run the include the following commands. Each of which will create a unique sweep ID, that can be used to run the first scenario with a specific algorithm.
```
wandb sweep sweeps/scenario_1/scenario_2_rl_agent.yaml # RL - Agent
wandb sweep sweeps/scenario_1/scenario_2_rl_expert.yaml # RL - Expert
wandb sweep sweeps/scenario_1/scenario_2_ad_agent.yaml # AD - Agent
wandb sweep sweeps/scenario_1/scenario_2_ad_expert.yaml # AD - Expert
wandb sweep sweeps/scenario_1/scenario_2_a2d_agent.yaml # A2D
```
Running any one of these commands will generate something similar to the following:
```
wandb: Creating sweep from: sweeps/scenario_2/scenario_2_<algo-key>.yaml
wandb: Created sweep with ID: <unique-id>
wandb: View sweep at: https://wandb.ai/<wandb-account>/scenario_1_a2d_ICML/sweeps/<unique-id>
wandb: Run sweep agent with: wandb agent /<wandb-account>/scenario_1_a2d_ICML/<unique-id>
```
In order to run this set of parameters, and log the evaluation out to weights and biases, you need to use the singularity container associated with this set of runs. To use this container, make sure that the "--nv" and "-B" are included or the container will likely not run correctly.
```
singularity run --nv -B $(pwd) a2d.sif wandb agent --count 1 <wandb-account>/scenario_2_a2d_ICML/<unique-id>
```
Again, programs can be run more manually by directly calling on the main function:
```
singularity run --nv -B $(pwd) a2d.sif python main.py --AD_batch_size=64 --AD_buffer_mem=25000 --AD_full_mem=1 --AD_updates_per_batch=25 --algo_key=a2d-agent --beta=0 --clipped_nominal_reward=0 --completed_reward=1 --critic_batch_size=64 --critic_updates=3 --delayed_policy_update=1 --entropy_coeff=0.003 --env_name=plai-carla/OvertakingTruck-v0 --eps=8.5e-05 --eval_interval=25000 --expert_frame_stack=1 --expert_reward=0 --force_fps=20 --gae_lambda=0.95 --gamma=0.995 --group=a2d_agent_eval --log_dir=./trained_models/scenario_2/a2d-agent-eval/ --log_interval=5 --log_lr=-4.5 --logged_moving_avg=25 --max_grad_norm=1.25 --max_time_horizon=300 --nominal_penalty=0 --norm_states=1 --num_env_steps=3000000 --num_processes=4 --num_steps=512 --policy_batch_size=512 --policy_updates=5 --pretrain_critic_updates=1 --save_dir=./trained_models/scenario_2/a2d-agent-eval/ --save_interval=1000 --seed=1 --survive_reward=1 --use_compressed_state=1 --use_gae=1 --use_linear_lr_decay=0 --use_log_lr=1 --value_loss_coeff=1.25 --waypoint_reward=0
```