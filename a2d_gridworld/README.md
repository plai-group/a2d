# Adaptive Asymmetric DAgger (A2D) Root Repository

This repository contains the source code for the gridworld experiments presented in the paper Robust Asymmetric Learning in POMDPs, Warrington, A.\*, Lavington, J. W.\*, Ścibior, A., Schmidt, M., & Wood, F. (2021). Robust Asymmetric Learning in POMDPs. _International Conference on Machine Learning, 2021.  arXiv preprint arXiv:2012.15566_. ([paper](https://arxiv.org/pdf/2012.15566.pdf)).


##### Contents:
- [Installation](#sec_ins)
- [TL;DR: I Want To Reproduce Results](#sec_rep)
- [Important Notes](#sec_not)
- [Using This Repository](#sec_use)
- [Contact](#sec_con)



## <a name="sec_ins"></a> Installation
This repository is relatively straightforward.  If directly cloning this repository then run:

```
git clone git@github.com:andrewwarrington/a2d_gridworld.git
cd a2d_gridworld
pip3 install -r requirements.txt
```

The repository is only dependent on pure python modules, and the only slightly awkward dependences that are required are PyTorch and OpenAI Gym, both of which can now be installed through pip.  



## <a name="sec_rep"></a> TL;DR: I Want To Reproduce Results
The code is configured for reproducing the experimental results in the paper with minimal user interaction or modification.  There are four main sets of experiments presented in the papers: Figure 4 & 5, Figure B.1, Figure B.2, and Figure B.3.  The current default parameter settings are as were used for the original experiments.  


#### Runnning on SLURM cluster:
Each experiment was run with at least ten independent repeats, and so full reproduction of results locally is somewhat infeasible.  To repeat each experiment, there is an "outer" and an "inner" bash/SLURM script.  The inner script is prefixed with an underscore, and will be repeatedly called by the outer script.  The inner SLURM script may (/will) need modifying to make it compatible with your SLURM resources.  Each repeat will run ALL methods (i.e. _RL(MDP)_, _RL(Asym)_, _A2D(Compact)_ etc).  To run all repeats on a distributed SLURM cluster:


##### Figures 4 & 5: Main gridworld experiments: 
```
# Run from a2d_gridworld 
./launch_scripts/launch_multiple.sh
```

The environment that is run is set in `./launch_scripts/_launch_multiple.sh` by changing the `env-name` parameters.  The values are `"MiniGrid-LavaGapS7-v0"` or `"MiniGrid-TigerDoorEnv-v0"` to run Frozen Lake (called lava-gap in the code...) and Tiger Door respectively.  With reasonable CPU/GPU resources, each repeat should run within about nine hours.  


##### Figure B.1: Investigating Q function:
```
# Run from a2d_gridworld 
./launch_scripts/launch_q.sh
```

The wrapper will launch the six experiments, (TD-1, TD-2, TD-3)x(Q, ~Q), for the prescripted number of repeats.  Each repeat and experiment will run on as an independent job, and dump in to the same folder.  For default settings, and with basic CPU/GPU resources, a single repeat should take much less than an hour to run.


##### Figure B.2: Investigating GAE parameter lambda.
```
# Run from a2d_gridworld 
./launch_scripts/launch_lambda_sweep.sh
```

Each value of lambda and each repeat will be run as a separate SLURM job, and will dump in to the same folder.  For default settings, and with basic CPU/GPU resources, a single repeat should take much less than an hour to run.


##### Figure B.3: Verifying performance under different representations. 
To run the RL on the partial state as well, set the flag `--rl-partial-state 1` in `_launch_multiple.sh`, and then run 

```
# Run from a2d_gridworld 
./launch_scripts/launch_multiple.sh
```

as before.  


####  Running locally:
To run individual experiments locally (for debugging, printing, investigating code etc), then call the Python file in `tests` directly.  The easiest script to call is:

```
python3 ./tests/A2D/RunA2DExperiments.py --env-name 'MiniGrid-TigerDoorEnv-v0'
```


or, to use the smaller TigerDoor-{1,2,3} environments with the default arguments from the supplement, then use:

```
python3 ./tests/A2D/RunQExperiments.py --env-name 'MiniGrid-TigerDoorEnv-v0'
``` 

where the `--env-name` flag sets the environment to run.  Individual (hyper)parameters/arguments can then be set using the standard `--`, or, directly in the Python file (if you are lazy, like me).


#### Visualising Results:
Each script generates a textual report in the log file (`report.txt`) and printed to `stdout` that can be quickly inspected.  For more involved plotting of multiple experiments and repeats, then there are a collection of plotting utilities and notebooks in the `plotting` directory.  These are documented in the notebooks themselves.  The notebooks _should_ have just one or two variables to update to set the filepath to the log directory you wish to plot, and should then generate the plots as appear in the paper.  (There is a heavy dose of _should_ in there, as minor changes to the workflow will seriously upset how data is parsed, read in and plotted.  If you are having trouble with it, then drop me an email on the address below. )



## <a name="sec_not"></a> Important Notes


#### Multiple Repositories?
While working remotely and in different timezones as a result of COVID-19, the long route to market for this paper, and one of the authors continuing to work on the project after the other had finished the degree program, the codebases used for the two main experiments in the paper (gridworld and autonomous vehicles (AV)) diverged quite significantly.  

Although the core of each algorithm is the same, there are several differences in the low-level implementation that make combining the codebases for exact reproduction of experimental results (tied to the exact Git commit or Weights and Biases experimental log) (a) an onerous task, and (b) would make the code unnecessarily complex and hard to parse.  

While it is on the to-do list to reconcile these codebases, the most sensible thing to do in the immediate future is to simply release both codebases verbatim.  This ensures that the results in the paper are immediately reproducible, and the _exact_ code used to generate those results can be inspected, critiqued, and built upon.  Each codebase can also stripped back for specifically that individual experiment, making the code as readily understandable as possible.  Finally, proposing bugfixes etc to each codebase individually reduces the chance that it _silently_ breaks the ability of the other codebase to reproduce results.  

In the future, when a unified codebase is created, it will be inserted directly into this repository.  The original repositories will still be linked as they currently are for indefinite preservation of experimental reproducibility and provenance.  


#### Should I Pull This Repository?
If you are only interested in reproducing the gridworld experiments, or you just want to look around a codebase to try and understand the algorithm, then pull this repository directly. 

There is a "parent" to this repository is [here](https://github.com/plai-group/a2d) that contains additional documentation, presentation materials, figures etc.  This gridworld repository is contained within the parent repository as a Git submodule, along with the AV repository.  If you want to reproduce all of the experimental results, browse the additional documentation, or look around the AV implementation, then pull the [parent repository](https://github.com/plai-group/a2d).  If you are only interested in the AV experiments and implementation, then head on over to the AV repository [here]().  

If you pull the parent repository, then all of the scripts in this submodule must be run from _within_ the submodule root; i.e. commands must start `./X.sh` as opposed to `./a2d_gridworld/X.sh` etc.


#### Important Note: Minigrid
This repository contains code from the `gym-minigrid` package, originally distributed by Maxime Chevalier-Boisvert Lucas Willems, and Suman Pal, available [here](https://github.com/maximecb/gym-minigrid). We had to apply non-trivial modifications to this code, and therefore we include this updated code directly in this repository.  Therefore, when citing/referencing this work, please also direct a citation/reference to the original authors.  The code was originally released under the Apache 2.0 license, and so I have included a change note in the python files that I have modified, and retained an original license notice at the top of each file. 



## <a name="sec_use"></a> Using This Repository
We now give a more involved description of the code contained.



#### Nomenclature & Language
There is a lot of stuff going on in here, and so we use some specific language to describe what is run.  This language is pretty much consistent throughout this repository, and so it is worth keying in to.  

We refer to a particular setting of a method and a representation as an "experiment", both discussed in more detail separately below.  Experiments include, for instance, _RL (MDP)_ and _A2D (Image)_.  Multiple different experiments may be tested in a single "run" or "repeat".  Experiments are run one after the other, and dump their results in to the same directory in `./tests/results/$ENV/$EXT`, where `$ENV` is the string name of the environment, and `$EXT` is the folder extension.  Each run is differentiated through an integer variable called `seed` (although actually seeding the random number generator is not implemented).  Each experiment creates a folder in `./tests/results/$ENV/$EXT` (i.e. `./tests/results/$ENV/$EXT/{rl_state, a2d_observe}`, and each repeat then generates a folder inside each experiment folder for each repeat of that experiment, i.e. `./tests/results/$ENV/$EXT/{rl_state, a2d_observe, ...}/{1, 2, 3, ...}`.  This means that when multiple repeats are used for a range of methods, they are all dumped in to the same folder, which is fully modular and can then be picked up and moved for plotting/preservation etc.  

Several different types of observations/representations can be generated:

- `state`:  complete/omniscient and low-dimensional MDP state representation.  For the gridworlds, this is typically a concatenation of multiple one-hot vectors describing the complete state of the environment.  In frozen lake, the position of the weak ice is included.  In TigerDoor, the position of goal and hazard are always visible.
- `partial_state`:  partial and low-dimensional POMDP representation.  The same as state, except that asymmetric information is not available.  In frozen lake, the location of the weak patch is not included, and so the observation is strictly the position of the agent.  In TigerDoor, the location of goal and hazard are only available after the button has been pushed.  
- `(partial_)observe`:  partial and high-dimensional POMDP representation.  The asymmetric information is (not)available as in `partial_state`.  The image is a 42x42x3 RGB image.  See Figure 3 of the paper.  Sometimes `partial` is prepended to differentiate it from `full_observe`, but most often `partial = partial_observe`.
- `full_observe`:  complete and high-dimensional MDP representation.  This is a bit of an unusual representation, and is mainly used for debugging code.  The asymmetric information is always available (and hence is operating in the complete information MDP), but is embedded in a high-dimensional image.  This is most useful for confirming that the code for processing the images isn't accidentally transposing/flipping the image, and that the policies/encoders are capable of learning the most expressive transform required.  

Several different methods are available:

- `rl`: Vanilla reinforcement learning.
- `d`: Symmetric imitation learning, which is implemented as DAgger.  For testing purposes only, really. 
- `ad`: Asymmetric imitation learning, which is implemented as asymmetric DAgger.
- `a2d`: Adaptive asymmetric DAgger.
- `arl`: Asymmetric RL, where the value function is conditioned on the true state, whereas the policy is conditioned on an observe.
- `ete`/`PreEnc`: Use a pretrained encoder learned offline using rollouts from under the MDP, that is then frozen and just the policy head is learned. 

A particular experiment is then described by a method and a representation.  I.e. `rl_state` corresponds to RL in the compact state representations.  

In the code, we refer to `expert` and `learner`.  The expert is always conditioned on `state`.  When performing _RL(MDP)_, the policy is learned in the expert slot.  When doing D/AD/AIL, the pretrained expert is loaded into `expert` and is used to supervise training.  In `ete`/`PreEnc`, the pretrained expert is loaded into `expert` and is used to generate rollouts from which the encoder is learned.  In A2D, the expert is the policy on which RL updates are applied to using the importance weighted gradient update.  The `learner` is then conditioned on the (potentially) asymmetric representation available at runtime (i.e. `observe`).  The only exceptions to this are _RL(Asym)_ and _PreEnc_, where the learner value function is conditioned on the complete MDP representation (as per Pinto _et al._ [2017]).


#### Repository Structure
There are four main directories:

- `./a2d` :  Contains the core code for each method.  Called via one of the scripts in `./tests`. 
- `./tests` :  Contains the scripts for running experiments, although these are not configured as actual Python tests.  `./tests/A2D` contains the scripts for running experiments.  A second directory will appear as soon as an experiment is run, `./tests/results`, which will hold all experimental results and logs.  
- `./launch_scripts` :  Contains bash and SLURM scripts for running multiple different experiments on distributed clusters.
- `./plotting` : Contains Jupyter notebooks for plotting / visualising results.  

All scripts must be called from the root directory of the repository, i.e. `python ./tests/A2D/RunA2DExperiments.py` or `./launch_scripts/launch_q.sh`.  

The default way to run the code is to call `python ./tests/A2D/RunA2DExperiments.py`. 


#### a2d Directory
This directory contains most of the nuts and bolts of the code.  There are three directories and two (although really only one) python module.  

The `environments` directory contains the modified minigrid environment, along with two basic python modules for wrapping gym environments.  Unless you really need to get down-and-dirty because you are using/modifying our code, there is not a massive amount of need to inspect the code here.

The `models` directory contains the network architectures used.  There is a separate file that contains the encoder module, that is then called in `models.py` when an encoder needs to be instantiated for either the policy or value network.  

The `util` directory contains much of the method-agnostic code.  This includes the code for performing the RL policy and value function updates (`rl_steps.py`, `rl_update.py`), performing rollouts (`sampler.py`), loading expert policies (`expert_management.py`), initialising policies (`inits.py`), a replay/data buffer (`replay_buffer.py`), and a number of general utility files (`torch_utils.py`, `helpers.py`).

There are then two python files: `A2D_base_class.py` and `A2Dagger.py`.  In practice, these are just a single python file split in to two.  `AdpAsymDagger` (defined in `A2Dagger.py`) is a child class of `A2D_base` (defined in `A2D_base_class`).  `A2D_base` simply defines all the length and tedious functionality that is common to all the experiments, such as evaluating performance and logging to file. `AdpAsymDagger` then contains more of the method-specific stuff.  I really struggle with long python files, especially when most of the code is largely irrelevant logging code, so I separated this functionality out.  If this is not to your taste, then you can simply copy `A2D_base` into `AdpAsymDagger`.


#### The A2D Class
The main body of the code is in the `AdpAsymDagger` class.  All experiments are run through this class.  There is a member function for each method (`A2D`, `AD`, `RL`, `ARL` and `PreEnc`).  Each of these functions calls the correct combination of functions to enact that method.  The available options are rolling out under expert, learner, or a mixture; applying an RL update to either expert or learner; applying a projection/AIL update to the learner; and then an optional logging step to evaluate all policies' performance.  RL updates breakdown further to be include a policy update and/or a value function update to the expert or learner.   These options are all performed inside the `step` function.

For instance, A2D is defined as: (1) rolling out under the mixture; (2) RL update to expert policy and expert value function; (3) RL update to learner value function; (4) projection update to learner; (5, optional) logging.

The RL update is implemented in `rl_update.py` and `rl_steps.py`.  The projection updates are then implemented in the A2D class. 


#### Run Scripts
In `tests` there are python files that start `Run{X}.py` and `{X}_Arguments.py`.  The `Run{X}.py` scripts first call the `{X}_Arguments.py` file, which contain all the parameters, hyperparameters, settings etc.  For each method, there is then a bespoke launching script that sets up the arguments, and calls the A2Dagger object to initiate training.  The bespoke launching scripts mainly do things like loading the appropriate expert, setting file paths, inscribing hyperparameters etc.  Each experiment is called in sequence. 


#### Logging
The performance is logged at regular intervals.  This is done through the member function `logging`.  This will perform a rollout for the expert and learner policy, using both the stochastic and deterministic version of the policy.  If a policy is not present, then it is not tested.  If both policies are present, then the KL divergence between the policies will be evaluated.  All of this data is then written to the log file `results.csv` which is saved in to the log directory.  A info dict is also saved, containing various parameters, optimiser states, parameter etc, for inspection after the fact.  The git information is also logged so experiments can be tracked to a particular git commit.  It is important therefore to make sure that the working directory is clean before running any significant experiments.  



## <a name="sec_con"></a> Contact
If you have questions, comments, bugfixes, or ideas for collaborations and extensions, then we are all ears!  Please reach out to us at `andrew (dot) warrington (at) keble (dot) ox (dot) ac (dot) uk` and we would be more than happy to help out wherever we can! :) 

Big love, 

Andy & Wilder




<br>
--- 

© Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, Andrew Warrington & J. Wilder Lavington.

<center>
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/logos/oxford.png" width="20%"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/logos/ubc.png" width="28%"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/logos/iai.png" width="27%"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/logos/plai.png" width="9%">
</center>
---










