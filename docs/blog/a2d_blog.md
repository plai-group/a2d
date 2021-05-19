<!-- Load LaTeX compiler using MathJax. -->
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"> 
</script>
<!-- -->   
 
add back \<video autoplay playsinline... 
<figure class="video_container">
  <video playsinline style="pointer-events: none;" allowfullscreen="true" width="100%" allowfullscreen="true" loop="true" muted="true">
    <source src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/occluded-pedestrian.mp4" type="video/mp4">
  </video>
</figure>
<!--<center>
<small>
<a name="fig1"></a> _Figure 1: The agent does not know there is an occluded hazard, and hence the policy is unsafe._ NOTE -- .
</small>
</center>--> 
 
# Robust Asymmetric Learning in POMDPs
<small>
Blog post by [Andrew Warrington](https://scholar.google.com/citations?user=MDj3OS4AAAAJ&hl=en) & [J. Wilder Lavington](https://scholar.google.com/citations?user=Ae2Qc0gAAAAJ&hl=en).  Full paper available [here](https://arxiv.org/pdf/2012.15566.pdf).  Talk available [here](https://github.com/plai-group/a2d/blob/master/docs/talk).  Code available [here](https://github.com/plai-group/a2d).
</small>



Imitation learning is great, allowing efficient and low-variance supervised learning to be exploited to transfer and manipulate policies.  Asymmetric information is also great, allowing for any additional information available at train time to be leveraged to expedite and stabilize training, while still producing a useable policy at deployment time.  

However, and maybe surprisingly, using imitation learning in conjunction with asymmetric information may lead to catastrophic results.  Because, put simply, the expert policy doesn't know what the agent policy does or doesn't know.  Therefore, the expert policy may force the agent into taking actions that are unsafe in the absense of the additional information.

We explore this fail-case in our recent ICML publication _Robust Asymmetric Learning in POMDPs_ [[war21a]](#war21a).  We then propose an algorithm, Adaptive Asymmetric DAgger (A2D), that allows the expert to be updated (or learned from scratch!) during imitation learning such that the agent learned using 

In this blog post, we will explore and expand on some of the ideas presented in the paper.  We won't go through all of the mathematics that underpins the method in excruciating detail here, instead focusing on the high level intuitions of the method.  For the detailed analysis we refer the reader to the main paper and supplementary materials [here](https://arxiv.org/pdf/2012.15566.pdf).









### Contents:
- [Background](#sec_for)
- [Deriving the Trainee Policy](#sec_tra)
- [Deriving the A2D Update](#sec_a2d)
- [A2D Algorithm](#sec_alg)
- [Results](#sec_res)
- [Conclusion \& Future Directions](#sec_con)

Other bits:

- [Code, Full Paper \& Talk](#sec_cod)
- [Glossary](#sec_glo)
- [Bibliography](#sec_bib)




## <a name="sec_for"></a> Background 

Reinforcement learning (RL) in Markov decision processes (MDPs) is a hard problem.  Stochastic environments, large action spaces, and long dependencies between actions and rewards make learning to optimally interact with an environment one of the most challenging problems in machine learning.  

This difficulty is exacerbated when the environment can only be partially observed.  From the perspective of the agent, taking identical actions when given identical observations may yield a dramatically different reward, due to parts of the environment that the agent cannot observe, or, could only observe a many steps ago.  This, effectively, increases the variability of the environment, and can require long histories of observations and actions to be considered when acting.  Furthermore, individual observations may be high dimensional, such as observing an image at each timestep.  

Together, these factors make the learning task exponentially more difficult, as the agent must also learn to effectively recover the pertinent information from the history, which may be embedded in tens of thousands of individual variables, while _also_ learning to interact with the environment. 

Therefore, an attractive alternative to direct RL, is to instead learn an agent (specifically referred to in this context as a _trainee_) by _imitating_ a policy that is capable of solving the task (referred to as an _expert_).  This is generally referred to as _imitation learning_ (IL) [[ros2011a]](#ros2011a).  This reduces RL to supervised learning, where the trainee is simply regressed onto the expert at each point in state space.  This separates learning to _percieve_ in the environment, from learning to _act_ in the environment, as the expert is already capable of acting.  This is especially attractive in partially observed environments where the observations are high-dimensional.  In such scenarios, imitation learning is observed to vastly outperform RL (i.e. [[ros2011]](#ros2011a)), both in terms of the number of environment interactions required to learn the policy, and the stability of learning.  

However, there are two major drawbacks to imitation learning.  Foremost is that an expert capable of solving the task (or examples of state trajectories produced from such an agent) is required.  This means that IL is not "end-to-end" -- that is to say, a performant agent can be learned from scratch with no substantial user input.  

In many simulated or controlled environments, we can obtain _asymmetric information_ at training time.  This means that when we are training the agent we can obtain information that is not available at deployment time, or, readily obtain the information in a more efficient representation.  For instance, at deployment time, the agent may only be able to observe a noisy, high-dimensional and partial RGB video feed.  However, at train time, we can obtain a complete (also referred to as fully observed or _omniscient_), low-dimensional state vector defining the entire state of the environment, i.e. everything is observed and is low-dimensional.  

Therefore, we can remedy the first IL drawback by using asymmetric RL to efficiently learn an expert.  We can then use this expert in asymmetric imitation learning to learn the trainee -- all for a lower comutational cost than if we had directly performed RL in the POMDP.

<br>
<center>
<a name="fig1"></a> 
![Simple comparison of DAgger and A2D.](/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/banner.png)

<small>
_**Figure 1**: High-level comparison of imitation learning (here we consider DAgger [[ros2011a]](#ros2011a)) and our algorithm, Adaptive Asymmetric DAgger (A2D)._ 
</small>
</center>
<br>






## <a name="sec_tra"></a> Imitation Learning \& Implicit Policies



## <a name="sec_a2d"></a> Deriving the A2D Update



## <a name="sec_alg"></a> A2D Algorithm
In [Algorithm 1](#alg1) we show the main A2D algorithm.  Crucially, we highlight, in blue, the two lines that extend AD to A2D.  


<center>
<a name="alg1"></a>
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/algorithm.png" width="40%">

<small>
_**Algorithm 1**: A2D algorithm.  Highlighted in blue are the lines that differentiate A2D from asymmetric DAgger (AD).  Note here we do not explicitly learn the Q-function.  To do this, Line 10 is converted to target the Q-function and is used to evaluate the reward-to-go in \\(\mathtt{RLSTEP}\\)._
</small>
</center>
<br>

Expanding on each line in the algorithm: 

**Line 3**: initialise the expert policy parameters, \\( \theta \\), trainee (variational) policy parameters, \\( \phi \\), expert value function parameters, \\( \nu\_{m} \\), and trainee value function parameters, \\( \nu\_{p} \\).

**Line 4**: Set \\( \beta = 1 \\) to start rolling out under the expert.  Initialise a data buffer, \\( D \\).

**Line 5**: For \\( N \\) training steps.

**Line 6**: Anneal \\( \beta \\).  \\( \beta \\) must tend to zero during training.

**Line 7**: Define the mixture policy, \\( \pi\_{\beta} \\), as the mixture of expert and trainee policies, with mixture coefficient \\( \beta \\).

**Line 8 & 9**: Gather data by rolling out under the mixture policy and add this data to the replay buffer.

**<span style="color:blue">Line 10</span>**: Parameterise the mixture value function as a mixture of the expert and trainee value functions.  

**<span style="color:blue">Line 11</span>**: Take the (importance weighted) RL step to update the value functions of the expert and trainee (backproping through the parameterization in Line 10), and the expert policy.  Advantages are importance weighted using the ratio in [EQU](#) which is easily computable.

**Line 12**: Take an asymmetric imitation learning step to update the (variational) trainee policy.


Therefore, A2D is a relatively easy-to-implement extension on top of a regular AIL implementation.  For some off-the-shelf RL policy gradient, the importance-weighted advantages may need to be computed upfront, since the stock implementations won't offer the required functionality.  Beyond this, the RL update step is pretty much default.



## <a name="sec_res"></a> Results
We explored our approach on two different environments: a gridworld environment, and an autonomous vehicles (AV) environment.  In all A2D experiments, the expert is learned simultaneously with the agent (i.e. the expert is learned from scratch).


### Gridworld
The first environments we study are two simple gridworld environments.  These are intended to study two particular failcases of asymmetric information.  In both scnearios, the agent (red) must navigate to the goal (green) while avoiding a hazard (dark blue).  Both scenarios are shown in [Figure 2](#fig2).


##### Scenario Description
The first scenario, referred to as Frozen Lake, explores the case where the agent is _never_ able learn something about the environment, and hence must adopt a different strategy.  In the MDP, the expert crosses the lake directly, stepping around the weak ice if it obstructs the agent.  In the POMDP, the agent should take the longer route, circumnavigating the lake.  

The second scenario, referred to as Tiger Door (adapted from [[find red]](#)), explores the case where the agent must take additional actions to uncover new information about the environment, whereas the expert would not need to perform these actions.  The agent is initially unaware if the position of the goal and tiger (hazard) are swapped.  The agent should therefore step onto the button (purple), which will reveal the configuration, and allow the agent to proceed to the goal position.

The agent is penalized for each action, and so circumnavigating the lake or de-touring via the button reduces the reward, but, removes the possibility of encountering the hazard, and the large negative reward that that incurs. 

In both scenarios, the agent observes the noisy, high-dimensional image shown in [Figure 2](#fig2), also referred to as "image" in [Figure 3](#fig3).  The expert observes a compact binary vector encoding the state of the environment (i.e. for Frozen Lake it is a 25-dimensional one-hot vector encoding the position of the agent concatenated with a 9-dimensional one-hot vector encoding the position of the weak ice).

<center>
<a name="fig2"></a>
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/gridworld/MiniGrid-LavaGapS7-v0_observe.png" width="20%"> &nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/gridworld/MiniGrid-LavaGapS7-v0_full_observe.png" width="20%"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/gridworld/MiniGrid-TigerDoorEnv-v0_observe.png" width="20%"> &nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/gridworld/MiniGrid-TigerDoorEnv-v0_full_observe.png" width="20%"> &nbsp;

<small>
_**Figure 2**: Gridworld environments.  **Left pair**: Frozen Lake.  **Right pair**: Tiger Door.  The left figure in each pair is the high-dimensional observation as seen by the agent.  The right figure in each pair shows the true underlying state.  In Frozen Lake, the agent (red) can never reveal the location of the weak ice (dark blue), and the location of the goal (green) is fixed.  The weak ice can be in any of the interior nine squares, and its location is never revealed to the agent.  In Tiger Door, the agent can step on the button (purple) to reveal the location of the goal and the (dark blue...?_ 🤔_) tiger, essentially allowing the agent to see the rightmost figure.  The tiger and goal are swapped from the shown configuration 50% of the time._
</small>
</center>
<br>


##### Results
We show the results for application of our method and several comparable methods in [Figure 3](#fig3).  We show the median and quartiles.  The optimal performance under full information and partial information are shown as _MDP_ and _POMDP_ respectively.  _RL (MDP)_ shows the performance of TRPO [[get ref]](#) in the MDP, which is able to quickly, efficiently and reliably obtain the optimal performance.  We then compare this to pure RL in the POMDP (_RL_), using separate policy and value networks, both conditioned on the image-based input, and asymmetric RL (_RL (Asym)_), where the policy is still conditioned on the image, but the value function is conditioned on the omniscient compact state [[pin2017a]](#pin2017a).  We see that both of these converge slowly, and, are very computationally demanding.  We then show asymmetric imitation learning (_AIL_), and see that this converges almost immediately, but converges to a poor solution.  We start this curve at 80,000 samples, as this is the upfront cost of training the expert, taken to be roughly the number of interactions at which _RL (MDP)_ had converged.  We also compare to using a pre-learned encoder (_Pre-Enc_).  This was learned using rollouts taken under the expert, converting the image into a compact (partial) state encoding.  We see that this is able to recover a good solution in a time commensurate with the _RL (MDP)_, in Frozen Lake, but again requires a number of pretraining examples.  In Tiger Door, this solution is poor as the MDP never pushes the button, and hence the correct required observations are never generated.

We then show our A2D method, when the trainee is provided with a compact-but-partial state encoding (_A2D (Compact)_, i.e. the perfect encoding) and when the trainee is provided with the raw image (_A2D (Image)_).  We see that these methods converge reliably to the optimal partially observing reward.  Crucially, these converge in a sample budget similar to one another.  This means that the imitation learning is successfully projecting the expert policy onto the high-dimensional trainee policy, and, that A2D is indeed able to recover the optimal partially observing policy.  

<br>
<center>
<a name="fig3"></a>
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/gridworld/sec4_results_IceLake_True_MiniGrid-LavaGapS7-v0_arXiv_final_2020_09_01__11_57_59_.png" width="40%"> &nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/gridworld/sec4_results_TigerDoor_True_MiniGrid-TigerDoorEnv-v0_arXiv_final_2020_09_01__11_57_29_.png" width="40%"> &nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/gridworld/legend_sec4_results_IceLake_True_MiniGrid-LavaGapS7-v0_arXiv_final_2020_09_01__11_57_59_.png" width="16.5%">

<small>
_**Figure 3**: Results for the two Gridworld experiments.  **Left**: Frozen Lake.  **Right**: Tiger Door. Results are normalized such that the optimal reward obtainable in the MDP is -1.0 (= -10<sup>0</sup>)._
</small>
</center>
<br>


We also show in [Figure 4](#fig4) the KL-divergence between expert and trainee during learning, for AIL and for A2D with both compact and image-based observations.  We see that the divergence in AIL saturates quickly and to a high divergence, indicating that the trainee is unable to replicate the expert.  We see in A2D that the divergence between the expert and trainee policy is initially low, increases as the expert learns to solve the MDP, and then reduces to a low divergence (note the log-axis) as the expert is forced in to solve the POMDP in a manner such that the trainee also solves the POMDP, resulting in a low final divergence.  We also see that the convergence paths are commensurate for both compact and image-based representations.  This again suggests that the imitation learning is successfully ameloriating the dimensionality and allowing compact and image-based policies to be learned in with similar dynamics.

<br>
<center>
<a name="fig4"></a>
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/gridworld/sec4_divergence_IceLake_True_MiniGrid-LavaGapS7-v0_arXiv_final_2020_09_01__11_57_59_.png" width="40%"> &nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/gridworld/sec4_divergence_TigerDoor_True_MiniGrid-TigerDoorEnv-v0_arXiv_final_2020_09_01__11_57_29_.png" width="40%">

<small>
_**Figure 4**: Divergence results for the two Gridworld experiments.  **Left**: Frozen Lake.  **Right**: Tiger Door.  Note the log-scale on the y-axis._
</small>
</center>
<br>



##### Q-Functions
In the supplement, we also explore the impact that directly learning the Q-function has on A2D.  We find that in certain environments, the Q-function must be learned (as opposed to using Monte Carlo rollouts) for A2D to function correctly.  Although this is both a fascinating and challenging quirk, we will not go in to this in detail here, and refer the reader to the supplement.  The key take-home is that to ensure A2D works correctly, the Q-function must be explicitly learned.  However, learning the Q-function increases the computational cost, number of hyperparameters that must be tuned, can introduce bias, and is observed to increase the number of environment interactions required to learn the complete the task.   We also note that while we were able to identify the root cause of this behaviour, we were unable to succinctly define a "test" or "condition" for if a particular environment requires learning a Q-function.  This is an interesting topic for further study.



### Autonomous Vehicles
The second environment we study is an autonomous vehicles (AV) scenario.  We use the AV simulator CARLA [[get ref]](#), and derive this scenario from one of the CARLA Challenge environments [[get ref]](#).  

##### Scenario


##### Results






## <a name="sec_con"></a> Conclusion \& Future Directions
In this work we looked at modifying an asymmetric imitation learning algorithm, such that the expert is modified, or learned online, with the trainee, such that the expert is learned to maximize the reward of the trainee after imitation.  This ameloriates the inherent drawbacks of reinforcement learning in high dimensions.  

In this work we also considered the scenario where the expert is an omniscient policy operating on the complete state representation of the underlying MDP.  An interesting alternative approach is the scenario where, instead of observing a complete state, the expert also only observes a partial state.  That may represent a more generalizable use-case, where expensive and accurate sensors can be used at training time, whereas only cheap, noisy, or incomplete sensor suites are available at test time.  Analysing this scenario is markedly more difficult as one must also reason about information that may be available to the agent, but is not available to the expert.

Beyond this, this work can be viewed as embedding a supervised learning or projection step _inside_ a policy gradient algorithm.  This presents a number of opportunities for extending our method to consider "optimizing" the agent such that the <span style="color:red">review whatever the reviewer was talking about</span>.

<span style="color:red">i want to talk something about how A2D is just a base and can be extended in a litany of ways</span> However, beyond these more concrete points, we believe this general approach is a new and exciting twist and combination of themes from across RL.  Our work is not so much about the particular implementation choices we made, rather that it is possible to pass an importance weighted policy gradient through an imitation learning procedure to update the expert policy, and that is can be cast as embedding a supervised learning task inside an RL task.  

If you are interested in building on this general idea, then please reach out!  We are always keen to talk science and build on our work, and combine it in new ways with other methods, or, apply it to hard problem domains! 

Big love, Andy & Wilder

<br>
---
</br>


## <a name="sec_cod"></a> Code, Full Paper \& Talk
Source code for reproduction of experimental results is available [here](https://github.com/plai-group/a2d).  We are currently working on a more flexible and extensible code implementation for deploying A2D.

The full paper is available on arXiv [here](https://arxiv.org/pdf/2012.15566.pdf) and the GitHub repo for the paper is available [here](https://github.com/plai-group/a2d/blob/master/docs/paper).  

The oral presentation given at the conference is available [here](https://github.com/plai-group/a2d/blob/master/docs/talk), the poster [here](https://github.com/plai-group/a2d/blob/master/docs/poster), and slide deck [here](https://github.com/plai-group/a2d/blob/master/docs/talk).

We have released all of these resources under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).  This means you are free to reproduce and share any of the figures, text and source code, with attribution to [[war21a]](#war21a), for non-commerical purposes.  Happy days!  




## <a name="sec_glo"></a> Glossary

- **Trainee**: An agent that is learned using imitation learning.
- **Expert**: A policy that is capable of acting optimally in an environment.  
- **Imitation learning** (IL): Learning a policy by regressing onto a second policy (usually an expert).  
- **Compact state**: A complete and low-dimensional representation of the true state of the environment.  
- **Markov decision process** (MDP): See [[war21a, S2.1]](#war21a).  A decision process where the agent/policy can observe the true and complete state of the environment, i.e. there is nothing "unknown".
- **Partially observed Markov decision process** (POMDP): See [[war21a, S2.2]](#war21a).  Extends an MDP to the case where the agent can only observe a partial or noise-corrupted version of the true state.




## <a name="sec_bib"></a> Bibliography

\* Denotes equal contribution.

<a name="war21a"></a> [war21a] **Warrington, A.\*, Lavington, J. W.\*, Ścibior, A., Schmidt, M., & Wood, F. (2021). Robust Asymmetric Learning in POMDPs. _International Conference on Machine Learning 2021, arXiv preprint arXiv:2012.15566_.** ([paper](https://arxiv.org/pdf/2012.15566.pdf))

<a name="pin2017a"></a> [pin2017a] Pinto, L., Andrychowicz, M., Welinder, P., Zaremba, W., & Abbeel, P. (2017). Asymmetric actor critic for image-based robot learning. _Robotics: Science and Systems XIV, arXiv preprint arXiv:1710.06542_. ([paper](https://arxiv.org/pdf/1710.06542.pdf))

<a name="ros2011a"></a> [ros2011a] Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. _Proceedings of the fourteenth international conference on artificial intelligence and statistics (pp. 627-635). JMLR Workshop and Conference Proceedings_. ([paper](http://proceedings.mlr.press/v15/ross11a/ross11a.pdf))


<br>
---



© Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, Andrew Warrington & J. Wilder Lavington.

<center>
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/logos/oxford.png" width="30%"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/logos/ubc.png" width="40%"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/logos/plai.png" width="11%">
</center>
---
