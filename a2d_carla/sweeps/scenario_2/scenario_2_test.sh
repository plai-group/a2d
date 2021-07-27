# generate sweeps
wandb sweep plotting/scenario_2/config_files/scenario_2_a2d.yaml
wandb sweep plotting/scenario_2/config_files/scenario_2_ad_agent.yaml
wandb sweep plotting/scenario_2/config_files/scenario_2_ad_expert.yaml
wandb sweep plotting/scenario_2/config_files/scenario_2_rl_agent.yaml
wandb sweep plotting/scenario_2/config_files/scenario_2_rl_expert.yaml

#
python sweep_runner.py --command 'singularity run --nv a2d.sif wandb agent --count 1 wilderlavington/scenario_2_a2d_ICML/gl8b4vet' \
--directory /ubc/cs/research/plai-scratch/wlaving/AdaptiveAsymmetricDAgger --machine borg --job_name a2d-agent-s2
python sweep_runner.py --command 'singularity run --nv a2d.sif wandb agent --count 1 wilderlavington/scenario_2_a2d_ICML/nh6jh1a3' \
--directory /ubc/cs/research/plai-scratch/wlaving/AdaptiveAsymmetricDAgger --machine borg --job_name rl-expert-s2
