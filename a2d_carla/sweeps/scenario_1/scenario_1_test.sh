
# generate sweeps
wandb sweep plotting/scenario_1/config_files/scenario_1_a2d.yaml
wandb sweep plotting/scenario_1/config_files/scenario_1_ad_agent.yaml
wandb sweep plotting/scenario_1/config_files/scenario_1_ad_expert.yaml
wandb sweep plotting/scenario_1/config_files/scenario_1_rl_agent.yaml
wandb sweep plotting/scenario_1/config_files/scenario_1_rl_expert.yaml

# test sweep
wandb sweep plotting/scenario_1/config_files/scenario_1_test.yaml

# run all the sweeps
python helpers/sweep_runner.py --command 'singularity run --nv -B $(pwd) a2d_new.sif wandb agent --count 1 iai/scenario_1_a2d_ICML/ddnicuu7' \
--directory /ubc/cs/research/plai-scratch/wlaving/AdaptiveAsymmetricDAgger --machine borg --job_name a2d-agent-s1
# ad
python helpers/sweep_runner.py --command 'singularity run --nv -B $(pwd) a2d.sif wandb agent --count 1 iai/scenario_1_a2d_ICML/erc7fyqf' \
--directory /ubc/cs/research/plai-scratch/wlaving/AdaptiveAsymmetricDAgger --machine borg --job_name ad-agent-s1
python helpers/sweep_runner.py --command 'singularity run --nv -B $(pwd) a2d.sif wandb agent --count 1 iai/scenario_1_a2d_ICML/rpzl8o8p' \
--directory /ubc/cs/research/plai-scratch/wlaving/AdaptiveAsymmetricDAgger --machine borg --job_name ad-expert-s1
# rl
python helpers/sweep_runner.py --command 'singularity run --nv -B $(pwd) a2d.sif wandb agent --count 1 iai/scenario_1_a2d_ICML/doaif2by' \
--directory /ubc/cs/research/plai-scratch/wlaving/AdaptiveAsymmetricDAgger --machine borg --job_name rl-agent-s1
python helpers/sweep_runner.py --command 'singularity run --nv -B $(pwd) a2d.sif wandb agent --count 1 iai/scenario_1_a2d_ICML/t7obxl4m' \
--directory /ubc/cs/research/plai-scratch/wlaving/AdaptiveAsymmetricDAgger --machine borg --job_name rl-expert-s1
# test
python helpers/sweep_runner.py --command 'singularity run --nv -B $(pwd) a2d.sif wandb agent --count 1 iai/scenario_1_a2d_ICML/dlceob6z' \
--directory /ubc/cs/research/plai-scratch/wlaving/AdaptiveAsymmetricDAgger --machine borg --job_name program-test-s1
