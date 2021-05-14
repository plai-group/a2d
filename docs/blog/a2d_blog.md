<!-- Load LaTeX compiler using MathJax. -->
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>
<!-- -->    

# Robust Asymmetric Learning in POMDPs
<small>
Blog post by [Andrew Warrington](https://scholar.google.com/citations?user=MDj3OS4AAAAJ&hl=en) \& [J. Wilder Lavington](https://scholar.google.com/citations?user=Ae2Qc0gAAAAJ&hl=en).  Full paper available [here](https://arxiv.org/pdf/2012.15566.pdf).  Talk available [here](https://github.com/plai-group/a2d/blob/master/docs/talk).  Code available [here](https://github.com/plai-group/a2d).
</small>

Reinforcement learning (RL) in Markov decision processes (MDPs) is a hard problem.  Stochastic environments, large action spaces, and long dependencies between actions and rewards make learning to optimally interact with an environment one of the most challenging problems in machine learning.  

This difficulty is exacerbated when the environment can only be partially observed.  From the perspective of the agent, taking identical actions when given identical observations may yield a dramatically different reward, due to parts of the environment that the agent cannot observe, or, could only observe a many steps ago.  This, effectively, increases the variability of the environment, and can require long histories of observations and actions to be considered when acting.  Furthermore, individual observations may be high dimensional, such as observing an image at each timestep.  

Together, these make the learning task exponentially more difficult, as the agent must also learn to effectively recover the pertinent information from the history, which may be embedded in tens of thousands of individual variables, while _also_ learning to interact with the environment. 

Therefore, an attractive alternative to direct RL, especially in partially observed environments, is to learn an agent (specifically referred to in this context as a _trainee_) by _imitating_ a policy that is capable of solving the task (referrd to as an _expert_).  This is generally referred to as _imitation learning_ (IL) [[ros2011a]](#ros2011a).  This reduces RL to supervised learning, where the trainee is simply regressed onto the expert at each point in state space.  This separates learning to _percieve_ in the environment, from learning to _act_ in the environment, as the expert is already capable of acting.  Imitation learning (somewhat unsurprisingly) is observed to vastly outperform RL (i.e. [[ros2011]](#ros2011a)), both in terms of the number of environment interactions required to learn the policy, and the stability of learning.  



<figure class="video_container">
  <video controls="true" allowfullscreen="true" width="100%" frameborder="0" allowfullscreen="true" loop="true" muted="true">
    <source src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/occluded-pedestrian.mp4" type="video/mp4" autoplay="false">
  </video>
</figure>
<center>
<small>
<a name="fig1"></a> _Figure 1: The agent does not know there is an occluded hazard, and hence the policy is unsafe._ NOTE -- NEED TO TURN AUTOPLAY BACK ON.
</small>
</center>


We tackle these problems in our recent ICML publication _Robust Asymmetric Learning in POMDPs_ [[war21a]](#war21a).  In this blog post, we will explore and expand on some of the ideas presented in the paper. 

### Contents:
- [Problem formulation](#sec_for)
- [Deriving the trainee policy](#sec_tra)
- [Deriving the A2D update](#sec_a2d)
- [Results](#sec_res)
- [Conclusion \& future directions](#sec_con)
- [Code, full paper \& talk](#sec_cod)
- [Glossary](#sec_glo)
- [Bibliography](#sec_bib)



## <a name="sec_for"></a> Problem formulation


## <a name="sec_tra"></a> Deriving the trainee policy


## <a name="sec_a2d"></a> Deriving the A2D update


## <a name="sec_res"></a> Results


## <a name="sec_con"></a> Conclusion \& future directions


## <a name="sec_cod"></a> Code, full paper \& talk
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

<a name="war21a"></a> [war21a] **Warrington, A.\*, Lavington, J. W.\*, Scibior, A., Schmidt, M., & Wood, F. (2021). Robust Asymmetric Learning in POMDPs. _International Conference on Machine Learning 2021, arXiv preprint arXiv:2012.15566_.** ([paper](https://arxiv.org/pdf/2012.15566.pdf))

<a name="ros2011a"></a> [ros2011a] Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics (pp. 627-635). JMLR Workshop and Conference Proceedings. ([paper](http://proceedings.mlr.press/v15/ross11a/ross11a.pdf))




<br>

© Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, Andrew Warrington & J. Wilder Lavington

<center>
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/logos/oxford.png" width="30%"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/logos/ubc.png" width="40%"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/blog/figures/logos/plai.png" width="11%">
</center>

