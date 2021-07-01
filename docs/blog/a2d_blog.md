<!-- Load LaTeX compiler using MathJax. -->
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"> 
</script>
<!-- -->   
 
<!-- add back \<video autoplay playsinline... 
<figure class="video_container">
  <video playsinline style="pointer-events: none;" allowfullscreen="true" width="100%" allowfullscreen="true" loop="true" muted="true">
    <source src="/Users/andrew/Documents/Public_repos/a2d/docs/figures/occluded-pedestrian.mp4" type="video/mp4">
  </video>
</figure> -->
<!--<center>
<small>
<a name="fig1"></a> _Figure 1: The agent does not know there is an occluded hazard, and hence the policy is unsafe._ NOTE -- .
</small>
</center>--> 
 
# Robust Asymmetric Learning in POMDPs
_Blog post by [Andrew Warrington](https://www.scholar.google.com/citations?user=MDj3OS4AAAAJ&hl=en) [Andrew Warrington](https://www.scholar.google.com/citations?user=MDj3OS4AAAAJ&hl=en) & [J. Wilder Lavington](https://www.scholar.google.com/citations?user=Ae2Qc0gAAAAJ&hl=en).  Full paper available [here](https://arxiv.org/pdf/2012.15566.pdf).  Talk available [here](https://www.github.com/plai-group/a2d/blob/master/docs/talk).  Code available [here](https://www.github.com/plai-group/a2d)._

<br>

Imitation learning (IL) is great.  IL reduces reinforcement learning (RL) to a supervised learning problem, regressing the _trainee_ policy onto an _expert_, where the expert is pre-trained and can already solve the environment.  This allows us to exploit efficient and low-variance supervision to transfer and manipulate policies, without the complexity and hassle of performing full reinforcement learning.  This is particularly beneficial in high-dimensional and partially observed environments, where jointly learning to perceive _and_ act is notoriously difficult.  The expert has already learned to act, and so the trainee only needs to learn to percieve such that it can replicate the actions of the expert. 

Asymmetric information in RL is also great, allowing any additional information available at train time to be leveraged to expedite and stabilize training, while still producing a policy that is not dependent on this asymmetric information, and hence is useable at deployment time.  Learning performant policies (or _experts_) conditioned on complete, asymmetric environments may be trivial.

However, and maybe surprisingly, using imitation learning in conjunction with asymmetric information may lead to catastrophic results.  This is because, put simply, the expert policy doesn't know what the agent (referred to as a _trainee_) policy does or doesn't know.  Therefore, the expert policy may force the trainee into taking actions that are unsafe in the absense of the additional asymmetric information.

We explore this unexplored theme in our recent ICML publication _Robust Asymmetric Learning in POMDPs_ [[war21a]](#war21a).  We demonstrate that while asymmetric IL (AIL) can vastly outperform RL in terms of computational complexity, it can converge to arbitrarily bad policies, as a result of the information asymmetry between trainee and expert.  We then propose an extension, Adaptive Asymmetric DAgger (A2D), that interleaves a tractable-to-compute policy gradient update, applied to the **expert** but maximizing the experted reward of the **trainee**, with regular AIL steps to then update the trainee policy conditioned on the updated expert.  We then demonstrate that A2D converges to the optimal trainee policy in scenarios where regular AIL would otherwise fail.  The A2D update is conceptually simple and is (relatively) easy to add to an existing AIL implementation.

In this post, we will explore and expand on some of the ideas presented in the paper.  We won't go through all of the mathematics that underpins the method in excruciating detail here, instead focusing on the high level intuitions of the method.  For the detailed analysis we refer the reader to the main paper and supplementary materials [here](https://arxiv.org/pdf/2012.15566.pdf).



##### Contents:
- [Background](#sec_for) (and a note on [notation](#sec_not)) 
- [Asymmetric Imitation Learning & Implicit Policies](#sec_tra)
- [Deriving the A2D Update](#sec_a2d)
- [A2D Algorithm](#sec_alg)
- [Results](#sec_res)
- [Discussion \& Future Directions](#sec_dis)

##### Other bits:

- [Code, Full Paper \& Talk](#sec_cod)
- [Glossary & Notation](#sec_glo)
- [Bibliography](#sec_bib)




## <a name="sec_for"></a> Background 

_If you are only interested in the technical content, this background section can be skipped._

Reinforcement learning (RL) in Markov decision processes (MDPs) is a hard problem.  Stochastic environments, large action spaces, and long dependencies between actions and rewards make learning to optimally interact with an environment one of the most challenging problems in machine learning.  

This difficulty is exacerbated when the environment can only be partially observed.  From the perspective of the agent, taking identical actions when given identical observations may yield a dramatically different reward, due to parts of the environment that the agent cannot observe, or, could only observe a many steps ago.  This, effectively, increases the variability of the environment, and can require long histories of observations and actions to be processed when acting.  Individual observations may be also high dimensional, such as observing an image at each timestep, further increasing the complexity of learning an optimal agent.  The agent must therefore also learn to effectively recover the pertinent information from the observation history, which may be embedded in tens of thousands of individual variables, while _also_ learning to interact with the environment.  Together, these factors make the learning task in a POMDP exponentially more difficult compared to learning in an MDP

Therefore, an attractive alternative to direct RL is to instead learn an agent (specifically referred to in this context as a _trainee_) by _imitating_ a policy that is capable of solving the task (referred to as an _expert_).  This is referred to as _imitation learning_ (IL) [[ros2011a]](#ros2011a).  As the expert is already capable of acting optimally, this separates learning to _percieve_ in the environment, from learning to _act_ in the environment.  This, in-turn, reduces learning a trainee to supervised learning, where the trainee policy is simply regressed onto the expert policy at each point in state space.  This is especially beneficial in partially observed environments, where histories of observations may be very high-dimensional.  In such scenarios, imitation learning is observed to vastly outperform RL (i.e. [[ros2011]](#ros2011a)), both in terms of the number of environment interactions required to learn the policy, and the stability of learning.  IL is shown schematically in [Figure 1](#fig1). 

However, there are two major drawbacks to imitation learning.  Foremost is that an expert capable of solving the task (or examples of state trajectories produced by such an expert) is required.  This means that IL is not "end-to-end" -- that is to say, a performant agent cannot be learned from scratch with no substantial user input as an expert (or trajectories) must be specified apriori.  Therefore, detractors will assert that learning an expert is as difficult as the original learning task.  

However, in many simulated or controlled environments, we can obtain additional information at training time that is not available to the agent at deployment time, referred to as _asymmetric information_.  Most often, asymmetric information corresponds to the _complete_ state of the environment, whereas the agent is only able to observe a part of the state (often referred to as _partial information_).  For instance, at deployment time, the agent may only be able to observe a noisy, high-dimensional, monocular RGB video feed.  However, at train time, we can obtain a complete (also referred to as fully observed or _omniscient_), low-dimensional state vector defining the entire state of the environment, i.e. everything is observed and in a low-dimensional representation.  A popular algorithm for exploiting this information, asymmetric actor-critic [[pin2017a]](#pin2017a), uses a value function conditioned on the asymmetric information to stabilize and expedite RL in a policy conditioned on the partial observations.

However, an alternative use-case is to simply learn a policy conditioned on just the asymmetric representation.  Due to the representational efficiency of the asymmetric information, this learning is easy and reliable.  We then leverage this policy as an expert for use in AIL to learn the partially observing trainee.  Therefore, we can remedy the first (A)IL drawback, that an expert must be provided, by using RL in the asymmetric domain to efficiently learn an expert, which is then projected onto a partially observing trainee policy -- all for a lower comutational cost than if we had directly performed RL in the POMDP.  

If the asymmetric information is the complete state representation (i.e. the MDP state), then this corresponds to performing RL in the MDP, and then projecting the solution on to a POMDP.  Note that this is the general context we will discuss: expert ↔ fully observing policy ↔ MDP; agent/trainee ↔ partially observing policy ↔ POMDP.  We return to comment on this in the [discussion](#sec_dis).

However, this general modus operandi has a significant and previously unremarked on drawback, and is the second flaw of AIL.  Imitation learning minimizes a divergence metric in the distribution over actions between the expert and agent policy over state space.  Crucially, this does not consider the _reward_ induced by each policy.  For instance, suppose the expert is optimal under the MDP.  The agent recovered by AIL is only _guaranteed_ to be optimal, _in terms of reward_, if the optimal POMDP policy is identical to the MDP policy.  This can be understood by considering the case where this does not hold: if the trainee has any probability of performing an action that the expert would not take, then this action may incur an infiniately large negative reward.  This is not addressed by minimizing a divergence, and so the learned policy may be arbitrarily poor, _unless_, the trainee is _identical_ to the expert.

This is a condition we refer to as _identifiability_ (Definition 2 in the [paper](https://arxiv.org/pdf/2012.15566.pdf)).  This is an _extremely_ strong condition, that is unlikely to hold in general environments.  Furthermore, identifiability is a property of two processes, as the optimal policy is (implicity) defined by the process itself (through the reward maximization).  Furthermore, it is not a practically testable condition (or rather a test would require both optimal policies, and hence the problem has already been solved).[<sup>1</sup>](#fn1)  Therefore, naively performing AIL in this way will either "work," or it won't.  When it does not, the consequences may be aribtrarily bad and not immediately apparent. 

Therefore, we wish to find a way to adapt the asymmetric expert such that it maximizes the reward of the trainee that imitates it.  This is the core intuition of our _adaptive asymmetric DAgger_ (A2D) algorithm.  Our algorithm adds an additional feedback loop that updates the expert during imitation learning to maximize the reward of the agent, shown in blue in [Figure 1](#fig1).  This update takes the form of an importance-weighted policy gradient update that we can pass through the imitation learning update.  Furthermore, we can actually simultaneously learn the expert and trainee, with no pre-trained elements (instead of learning the expert offline and then updating it).  Therefore, A2D can be considered as an approach to end-to-end RL that leverages and embedded projection step.  

In the next couple of sections we will briefly survey the core results that enable A2D.  For more detailed derivations and discussion we refer the reader to the [paper](#https://arxiv.org/pdf/2012.15566.pdf).


<center>
<a name="fig1"></a> 
![Simple comparison of DAgger and A2D.](/Users/andrew/Documents/Public_repos/a2d/docs/figures/banner.png)

<small>
_**Figure 1**: High-level comparison of imitation learning (here we consider DAgger [[ros2011a]](#ros2011a)), shown in black, and our algorithm, Adaptive Asymmetric DAgger (A2D), which adds an additional feedback loop to IL, shown in blue.  The additional feedback loop is implemented by lines 10 & 11 in [Algorithm 1](#alg1), also highlighted in blue._ 
</small>
</center>
<br>

<small>
<a name="fn1"></a>
_[<sup>1</sup>](#fn1)As an aside, identifibility implies that there exists a transformation of the observations that yields the underlying state.  We do not discuss this in the main paper, and hence an interesting topic for future work is attempting to define a more useable test for identifiability, as well as developing reasoning around the consequences of indentifiability when learning encoders._
</small>


### <a name="sec_not"></a> A Note on Notation...
We try to keep this post light on the math, however, to succinctly talk about the work, some notation and terminology is required.  We therefore recommend that the reader surveys Section 2 of the [paper](#https://arxiv.org/pdf/2012.15566.pdf), as this section contains all of the technical background and core definitions we will use throughout this work.  Many people familiar with RL in POMDPs will be very comfortable with most of the terminology we use.  We also include a table of notation in [Table A.1](#tab_a1).


## <a name="sec_tra"></a> Asymmetric Imitation Learning \& Implicit Policies
The first thing we must define is the asymptotic properties of asymmetric imitation learning.  Concretely defining this will then allow us to exploit these definitions to derive the A2D update.  This material is tackled in Section 3.

We define AIL as learning the parameters of the trainee policy, \\( \phi \\), by minimizing the expected divergence between the action distribution produced by the expert policy and agent policy across the joint occupancy distribution over state space and belief state space under the trainee:

<center>
\\(  \phi^* = \mathrm{argmin}\_{\phi \in \Phi} \ \mathbb{E}\_{s, b \sim d^{\pi \_{\eta}}(s,b)}  \left[ \mathbb{KL} \left[ \pi \_{\theta} (a|s) || \pi \_{\phi} (a|b) \right] \right]. \\)
</center>

The expert is conditioned on the full state, \\( s \\), whereas the trainee is conditioned on a belief state, \\( b \\).  The belief state is a representation of the history of observations and actions to that point [[mur2000a]](mur2000a) on which the partially observing trainee/agent is conditioned.  Throughout this work, we assume that the belief state is a rolling buffer of the last \\( w \\) observations and actions, often referred to as a windowed belief state.  The policy \\( \pi \_{\eta} \\) if referred to as a _behavioural policy_, and is a fixed policy under which you interact with the environment.  In later steps of this derivation, this policy will be set equal to the current trainee policy.  The term \\( d^{\pi_{\eta}}(s,b) \\) is then the joint distribution over states and belief states induced by the superscripted policy.

We show, in Theorem 1, that the solution of this minimization can be written as:

<center>
\\( \hat{\pi}^{\eta}_{\theta} (a | b) := \mathbb{E} \_{d^{\pi \_{\eta}} (s | b)} \left[ \pi _{\theta} (a | s) \right] \\) 
</center>

where we refer to the policy \\( \hat{\pi}^{\eta} \_{\theta} (a | b) \\) as the _implicit policy of_ \\( \pi \_{\theta} \\) _under_ \\( \pi \_{\eta} \\), and is defined formally in Definition 1.  Crucially, this allows us to concretely define the solution to the AIL objective as an expectation.  However, using the implicit policy directly is intractable, as sampling from the conditional occupancy, \\( d^{\pi \_{\eta}(s | b)} \\), is not generally tractable.[<sup>3</sup>](#fn2)

Fortunately, we can avoid this by learning a variational approximation to the implicit policy, denoted \\( \pi \_{\phi} (a | b) \\) by minimizing the following objective:

<center>
\\( \psi^* = \mathrm{argmin} \_{\psi \in \Psi} F(\psi) \\)

\\( F(\psi) = \mathop{\mathbb{E}} \_{d^{\pi\_{\eta}}(s,b)}  \left[ \mathbb{KL} \left[ \pi \_{\theta}(a|s) || \pi \_{\psi}(a|b) \right] \right], \\)

\\( \nabla \_{\psi} F (\psi) = \mathbb{-E} \_{d^{\pi \_{\eta}}(s, b)} \left[ \mathop{\mathbb{E}} \_{\pi_{\theta}(a | s)} \left[ \nabla _{\psi} \log \pi _{\psi} (a | b) \right] \right], \\)
</center>

where, assuming the variational family is sufficiently expressive, this divergence between the implicit policy and the variational approximation can be exactly minimized.  Crucially, we can compute the gradient of this objective using samples from the _joint occupancy_, \\( d^{\pi \_{\eta}}(s, b) \\), from which we can sample from (as opposed to the conditional occupancy).  This means it is tractable to learn a variational approximation using samples generated from rolling out under \\( \pi \_{\eta} \\).  

However, we require that \\( \pi \_{\eta} = \hat{\pi}^{\eta} \_{\theta} \\), such that the implicit policy (and hence the variational approximation) is optimal under the distribution over states and belief states induced _by that policy_.  This expression therefore defines a fixed point equation.  The policy that solves this fixed point equation is denoted simply as \\( \hat{\pi} \_{\theta} \\), and is referred to as just the _implicit policy of_ \\( \pi \_{\theta} \\).  

Therefore, to solve this fixed point equation, we define the following iteration:

<center>
\\( \psi \_{k+1} = \mathbb{argmin} \_{\psi \in \Psi} \mathop{\mathbb{E}} \_{d^{\pi \_{\psi \_k}}(s,b)}  \left[  \mathbb{KL} \left[ \pi \_{\theta}(a|s) || \pi \_\psi(a|b) \right] \right]. \\) 
</center>

It can be shown that this iteration converges to the optimal variational approximation of the implicit policy, by alternating between rolling out under the current policy, \\( \pi \_{\psi \_k} \\), and then regressing the trainee onto the expert at each point in these rollouts using the gradient defined above.  This is the basic loop that underpins imitation learning, and is simply supervised learning/regression using data gathered from a standard rollout under the policy.

However, for AIL to be guaranteed to yield an optimal policy, we require that \\( \pi \_{\psi \_k} \rightarrow \pi \_{\phi \^\*} \\) as \\( k \rightarrow \infty \\), where \\( {\phi \^\*} \\) is defined instead as the optimal partially observing policy under the RL objective, and where we are imitating an optimal expert policy, \\( \pi \_{\theta^*} \\).[<sup>3</sup>](#fn3)  

To explore when this holds, we first define the notion of _identifiable policy pairs_ (Definition 2).  An identifiable policy pair is an expert-trainee policy pair for which the divergence between the policies is zero:

<center>
\\( \mathop{\mathbb{E}} \_{d^{\pi\_{\psi}}(s,b)}  \left[ \mathbb{KL} \left[ \pi \_{\theta}(a|s) || \pi \_{\psi}(a|b) \right] \right] = 0. \\)
</center>

This means that the trainee is capable of exactly imitating the expert at every point visited.  We note that these conditions only need to hold in regions of state and belief-state space where the policy places mass.  This is a weaker condition than requiring that the policies are equal in _all_ of state space.  

This is then extended to _identifiable process pairs_ (Definition 3).  For processes to be identifiable, we require that all optimal fully observing policies, and their corresponding partially observing implicit policy, form identifiable policy pairs.  This means that for _any_ expert policy learned on the MDP, the POMDP policy learned through AIL is guaranteed to be optimal under partial information.  If this condition holds, we show in Theorem 2 that AIL is guaranteed to converge to the optimal partially observing policy.  

If processes are not identifiable, then the reward earned by the trainee may be arbitrarily sub-optimal.  This is the intuition that we take direction from.  The expert must therefore be updated such that it provides asymtotically unbiased supervision to the trainee, with respect to the optimal partially observing policy, as opposed to simply the implicit policy. 

We briefly survey this derivation in the next section.

<a name="fn2"></a>[<sup>2</sup>](#fn2) _To sample from this distribution would require one of the following: being able to force rollouts to go through a particular belief state; the belief state defining a predictive distribution over the true state; or; a method for generating an unbiased estimate of the true state from the observation history.  These are not simple problems, and hence we consider sampling from this distribution to be intractable._

<a name="fn3"></a>[<sup>3</sup>](#fn3) _We note that we are a little lax in notation here, and overload \\( \phi \^\* \\) to indicate the optimal agent policy under the RL_ and _under the AIL objective, even when these policies may not be the same.  This is to avoid adding more notational clutter.  Practically speaking however, \\( \phi \\) is/cannot be learned using the original AIL objective, and instead  \\( \psi \\) is learned, and so there is no clash practically speaking._




## <a name="sec_a2d"></a> Deriving the A2D Update
The full derivation for the A2D update is very involved.  We present the full derivation in the supplementary materials of the main paper.  However, the high-level overview of the derivation is (relatively) easy to follow.  We begin by deriving A2D in terms of Q functions.  We then modify A2D to use Monte Carlo rollouts, removing the need to directly estimate the Q function, but at the expense of introducing bias.  

##### Deriving A2D with Q Functions

We begin by defining the RL objective in terms of the Q function of the implicit policy:

<center>
\\( \theta^\* = \mathrm{argmax} \_{\theta \in \Theta} J(\theta), \\)

\\( J(\theta) = \mathop{\mathbb{E}} \_{d^{{\hat{\pi} \_{\theta}}}(b) \hat{\pi} \_\theta(a|b) } \left[ Q^{{\hat{\pi} \_{\theta}}}(a,b) \right], \\)


\\(\mathrm{where}\ \  Q^{{\hat{\pi} \_{\theta}}}(a,b) = \mathop{\mathbb{E}} \_{ p(b',s',s|a,b)} \left[ r(s,a,s') + \mathop{\gamma\mathbb{E}} \_{\hat{\pi} \_\theta(a'|b')} \left[ Q^{\hat{\pi} \_\theta} (a', b') \right] \right]. \\)
</center>

The Q function defines the expected reward ahead conditioned on a particular belief state and action.  This simple objective defines a complex nested optimzation:

<center>
\\( \psi = \mathrm{argmin} \_{\psi \in \Psi} \mathop{\mathbb{E}} \_{d^{\pi \_{\psi}}(s,b)} \left[ \mathbb{KL} \left[ \pi \_{\hat{\theta}^\*}(a|s) || \pi \_\psi(a|b) \right] \right], \\)

\\( \mathrm{where} \ \  \hat{\theta}^* = \mathrm{argmin} \_{\theta \in \Theta} \mathop{\mathbb{E}} \_{d^{{\pi \_{\psi}}}(b) \hat{\pi} \_{\theta}(a | b)} \left[Q^{{\hat{\pi} \_{\theta}}}(a,b) \right] \\)

</center>

We are minimizing the KL between the expert and trainee policies, where the expert policy we are targeting is defined by maximizing the reward over the implicit policy.  Since solving this nested optimization exactly is intractable, we first define an iterative procedure:

<center>
\\( \psi \_{k+1} = \mathrm{argmin} \_{\psi \in \Psi} \mathop{\mathbb{E}} \_{d^{\pi \_{\psi \_k}}(s,b)} \left[ \mathbb{KL} \left[ \pi \_{\hat{\theta}^\*}(a|s) || \pi \_\psi(a|b) \right] \right], \\)

\\( \mathrm{where} \ \  \hat{\theta}^* = \mathrm{argmin} \_{\theta \in \Theta} \mathop{\mathbb{E}} \_{\hat{\pi} \_{\theta}(a | b) d^{{\pi \_{\psi \_k}}}(b)} \left[Q^{{\hat{\pi} \_{\theta}}}(a,b) \right] \\)
</center>

The first minimization here is a standard AIL update, applied to the (variational) trainee policy, using the current expert policy.  However, to make this iteration tractable, and show that this iteration converges (under some assumptions) we must further unpack the second term.  

We first define a surrogate reward, \\( J \_{\psi} (\theta) \\), as a lower bound on the original objective:

<center>
\\( \max \_{\theta \in \Theta} J \_{\psi}(\theta) = \max \_{\theta \in \Theta} \mathop{\mathbb{E}} \_{{\hat{\pi} \_\theta(a|b) d^{{\pi \_{\psi}}}(b)}} \left[ Q^{{\pi \_{\psi}}}(a,b) \right] \\)

\\( \leq \max \_{\theta \in \Theta} J(\theta) = \max_{\theta \in \Theta}  \mathop{\mathbb{E}} \_{{\hat{\pi} \_{\theta} (a|b) d^{{\pi \_{\psi}}}(b)}} \left[ Q^{\hat{\pi} _{\theta}}(a,b) \right]. \\)
</center>

We refer the reader to Section C.3.1 to see why this surrogate object is a lower bound, where an extended discussion is presented.  The key take-away from this, however, is that we are able to replace the Q function of the implicit policy with the Q function of the current behavioural policy.  

We can then unpack the expectation by exploiting the closed-form definition of the implicit policy:

<center>
\\( \nabla \_{\theta} J \_{\psi}(\theta) =  \nabla \_{\theta} \mathop{\mathbb{E}} \_{{\hat{\pi} \_\theta(a|b) d^{{\pi \_{\psi}}}(b)}} \left[ Q^{{\pi \_{\psi}}}(a,b) \right] \\)

\\( \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad = \mathop{\mathbb{E}}\nolimits \_{{d^{\pi \_{\psi}}(b)}} \left[ \nabla \_{\theta} \mathop{\mathbb{E}} \_{{ d^{{\pi \_{\psi}}}(s | b)}} \left[  \mathop{\mathbb{E}} \_{{\pi \_\theta(a|s)}} \left[ Q^{{\pi \_{\psi}}}(a,b) \right] \right] \right] \\)


\\( \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad = \mathop{\mathbb{E}} \_{{d^{{\pi \_{\psi}}}(s, b)}} \left[ \mathop{\mathbb{E}} \_{\pi \_{\theta}(a|s)} \left[ Q^{{\pi \_{\psi}}}(a,b) \nabla \_\theta \log \pi \_{\theta}(a | s) \right] \right] \\)
    
<a name="equ_a1"></a>\\( \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad = \mathop{\mathbb{E}} \_{d^{\pi \_{\psi}}(s, b) \pi \_{\psi}(a | b)} \left[ \frac{\pi \_{\theta}(a|s)}{\pi \_{\psi}(a | b)} Q^{{\pi \_{\psi}}}(a,b) \nabla \_\theta \log \pi \_{\theta}(a | s) \right]. \quad \quad \mathrm{(a.1)} \\)
</center>

where we have applied a REINFORCE gradient estimator to go from line two to line three.  This term provides us with an estimator of the gradient of the parameters of the expert, with respect to the lower bound on the reward of the trainee, implicitly defined by the expert.  As we have formulated this objective in terms of the implicit policy, it is enforced that the supervision provided by the AIL step is "correct".  Furthermore, this requires that we roll out under the trainee itself.   Hence, we are gathering samples under the correct occupancy.  Note that during this rollout we collect _both_ state and belief state.  This gradient estimator is then intuitive in its form.  It is the standard REINFORCE gradient of the expert, but where the Q values are importance weighted, as we gathered data under the trainee and not the expert.  

To use A2D is in this form is straightforward.   We first rollout under the current trainee policy, \\( \pi \_{\psi \_k} \\).  We then (re-)fit a Q function approximator to these rollouts, \\( Q^{\pi \_{\psi \_k}} (a,b) \\).  We then importance weight the Q function evalautions at each samples state, belief state, action tuple, and apply a standard reinforce gradient update to the expert.  We then regress the trainee on to this updated expert using a standard AIL step.  We show in the supplement that under some assumptions, this form of A2D converges to the optimal POMDP policy.


##### Using Monte Carlo Rollouts
However, learning Q function is difficult, and the resulting algorithm is _highly_ dependent on the quality of the function approximation.  Therefore, many RL algorithms instead use Monte Carlo evaluations of the Q function, obtained directly from rollouts.  This increases the variance of the estimator, but reduces bias.  

Therefore, we defien a second A2D gradient estimator, this time in terms of Monte Carlo advantage estimates, computed using generalized advantage estimation (GAE) [sch2015a](sch2015a):

<center>
\\( \nabla \_\theta J \_{\psi}(\theta) = \mathop{\mathbb{E}} \_{\substack{d^{\pi \_{\psi}}(s \_t, b \_t) \\ \pi \_{\psi}(a \_t | s \_t, b \_t)}} \left[ \frac{\pi \_{\theta}(a \_t|s \_t) }{\pi \_{\psi}(a \_t | s \_t, b \_t)} \hat{A}^{\pi \_{\psi}} \nabla \_\theta \log \pi \_{\theta}(a | s) \right] ,  \\)

\\( \mathrm{where} \quad \hat{A}^{\pi \_{\psi}}(a \_t,s \_t,b \_t) = \sum \_{t=0}^{\infty} (\gamma \lambda)^t \delta_t ,  \\)

\\( \mathrm{and} \quad \delta \_t = r_t + \gamma V^{\pi \_{\psi}}( b \_{t+1}) - V^{\pi \_{\psi}}( b \_t) \\)
</center>

Here, \\( \hat{A}^{\pi \_{\psi}} \\) is the advantage as computed by GAE.  GAE mixes the Monte Carlo estimate of the sum of rewards ahead at each sampled point, with the lower-variance estimate computed by the value function.  This estimator can be directly substituted in in-place of the Q function based estimator, and instead only requires estimating the value function (a conceptually and practically easier task), and leverages the unbiased Monte Carlo Q function estimates.

However, and somwhat surprisingly, this estimator actually provides biased gradient estimates for A2D.  The exact root of this bias is exceptionally challenging to see.  However, inspecting the definition of \\( \delta_t \\), there is a dependence on \\( r_t \\).  Crucially, this reward term is conditioned on the _true state_, \\( s_t \\).  This means that by using a Monte Carlo rollout, this is _not_ an unbiased estimate.  Previously, in Equation [(a.1)](equ_a1), this dependency over state is _marginalized_ out by the Q function itself.  When using Q functions, the advantage computation is _entirely_ replaced by the Q function evaluation, and hence state plays no role.  In the Monte Carlo case however, this is not the case.  There is an extensive discussion of this phenomena in presented in the supplementary materials.

However, we believe that this bias, in most environments, is small.  By removing the dependency on Q functions, we also reduce the overall bias in the algorithm in practice by reducing the dependence on imperfect function approximators.  Furthermore, learning the Q function is computationally demanding and can be difficult to tune, and, if tuned poorly, can completely halt learning.  Therefore, we hence recommend that the Monte Carlo estimator is used.  As discussed further in the supplement, the bias in the estimator can be reduced by reducing the GAE parameter \\( \lambda \\).  Investigating more sophisticated methods for controlling this bias, either through controlling \\( \lambda \\), or, through the use of mixing in Q function estimation, is a promising avenue of future research.  

The final extension we provide, taking inspiration from the original DAgger algorithm [[ros2011a]](ros2011a), is that we allow rollouts to be gathered under a mixture of expert and trainee policies:

<center>
\\( \nabla \_\theta J \_{\psi}(\theta) = \mathop{\mathbb{E}} \_{\substack{d^{\pi \_{\beta}}(s \_t, b \_t) \\ \pi \_{\beta}(a \_t | s \_t, b \_t)}} \left[ \frac{\pi \_{\theta}(a \_t|s \_t) }{\pi \_{\beta}(a \_t | s \_t, b \_t)} \hat{A}^{\pi \_{\beta}} \nabla \_\theta \log \pi \_{\theta}(a | s) \right] ,  \\)

\\( \mathrm{where} \quad \hat{A}^{\pi \_{\beta}}(a \_t,s \_t,b \_t) = \sum \_{t=0}^{\infty} (\gamma \lambda)^t \delta_t ,  \\)

\\( \mathrm{and} \quad \delta \_t = r_t + \gamma V^{\pi \_{\beta}}( b \_{t+1}) - V^{\pi \_{\beta}}( b \_t), \\)

\\( \mathrm{and} \quad \pi^{\beta} (a \_t | s \_t , b \_t) = \beta \pi_{\psi} (a \_t |  b \_t) + (1 - \beta) \pi \_{\theta} (a \_t ). \\)
</center>

The mixture coefficient, \\( \beta \\), is then annealed from one to zero during training.  Practically, we parameterize the mixture value function as the \\(\beta\\)-mixture of the expert and trainee value functions.  This is an assumption, however, we note that the parameterization yields the correct value function once \\( \beta = 0 \\), and is computationally trivial to implement and enforce this behaviour.  However, we note that often best performance was obtained by setting \\(\beta=0\\) immediately.  

Monte Carlo A2D is then simple to use.  We first rollout under the current trainee policy, \\( \pi \_{\psi \_k} \\), or mixture policy \\( \pi \_{\beta \_k} \\).  We then (re-)fit the value function approximators to these rollouts, \\( V^{\pi \_{\psi \_k}} (b) \\) and \\( V^{\pi \_{\theta \_k}} (s) \\).  We then compute the GAE-smoothed advantage estimators, apply the importance weight at each sampled state-belief state-action tuple, and apply a standard reinforce gradient update to the expert.  We then regress the trainee on to this updated expert using a standard AIL step.  While this formulation is biased, we find practically that it converges more quickly and more reliably than when estimating the Q function.  


##### Summary
What we have shown, in short, is that we can pass a policy gradient _through_ the AIL operator.  This policy gradient maximizes the reward of the policy recovered by AIL with respect to the parameters of the expert policy.  This RL update is therefore applied in the low-dimensional, omniscient MDP space, and hence, in theory, is a lower-variance and cheaper to compute RL update. An AIL step then recoveres the trainee conditioned on the updated expert.  As we formulated this using the exact definition of the solution to the AIL objective, this is (under some assumptions) guaranteed to recover the optimal POMDP policy, and, with a zero KL divergence between expert and trainee.  Therefore, A2D _enforces_ identifiability of policies, even if the underlying processes are not identifiable.  

This gradient update is tractable to compute, only requiring an importance weighted Q value.  However, as learning Q functions can be difficult, we present a modification of the algorithm that uses Monte Carlo evaluations, value functions, and GAE.  Although this introduces bias into the estimator, we believe that this bias is small, and is rarely manifest in practice.  Any bias that is introduced is far outweighed by the complexity and difficulties in relying on Q function approximations.  

Finally, A2D requires _no_ pretrained artefacts.  The expert can be learned from scratch, online with the trainee.  Therefore, A2D represents an end-to-end method for training high-dimensional, partially observing trainees, while still exploiting the efficiency of AIL.



## <a name="sec_alg"></a> A2D Algorithm
In [Algorithm 1](#alg1) we show the main A2D algorithm.  Crucially, we highlight, in blue, the two lines that extend AD to A2D.  


<center>
<a name="alg1"></a>
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/figures/algorithm.png" width="40%">

<small>
_**Algorithm 1**: A2D algorithm.  Highlighted in blue are the lines that differentiate A2D from asymmetric imitation learning (AIL), such as asymmetric DAgger.  Note here we do not explicitly learn the Q-function.  To do this, Line 10 is converted to target the Q-function and is used to evaluate the reward-to-go in \\(\mathtt{RLSTEP}\\)._
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

**<span style="color:blue">Line 11</span>**: Take an RL step to update the value functions of the expert and trainee (noting that we do not backprop through the parameterization in Line 10, and instead target the value ahead independently).  We then compute the GAE-smoothed advantages, and importance weight, and then apply a standard REINFORCE RL update to the expert.

**Line 12**: Take an asymmetric imitation learning step to update the (variational) trainee policy.


Therefore, A2D is a relatively easy-to-implement extension on top of a regular AIL implementation.  For some off-the-shelf RL policy gradients implementations, the importance-weighted advantages may need to be computed upfront, since the stock implementations won't offer the required functionality.  Once the importance weighted advantages have been computed, the RL update step is unchanged.




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
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/figures/gridworld/MiniGrid-LavaGapS7-v0_observe.png" width="20%"> &nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/figures/gridworld/MiniGrid-LavaGapS7-v0_full_observe.png" width="20%"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/figures/gridworld/MiniGrid-TigerDoorEnv-v0_observe.png" width="20%"> &nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/figures/gridworld/MiniGrid-TigerDoorEnv-v0_full_observe.png" width="20%"> &nbsp;

<small>
_**Figure 2**: Gridworld environments.  **Left pair**: Frozen Lake.  **Right pair**: Tiger Door.  The left figure in each pair is the high-dimensional observation as seen by the agent.  The right figure in each pair shows the true underlying state.  In Frozen Lake, the agent (red) can never reveal the location of the weak ice (dark blue), and the location of the goal (green) is fixed.  The weak ice can be in any of the interior nine squares, and its location is never revealed to the agent.  In Tiger Door, the agent can step on the button (purple) to reveal the location of the goal and the (dark blue...?_ 🤔_) tiger, essentially allowing the agent to see the rightmost figure.  The tiger and goal are swapped from the shown configuration 50% of the time._
</small>
</center>
<br>


##### Results
We show the results for application of our method and several comparable methods in [Figure 3](#fig3).  We show the median and quartiles.  The optimal performance under full information and partial information are shown as _MDP_ and _POMDP_ respectively.  We see that in the POMDP (_RL_), using separate policy and value networks, both conditioned on the image-based input, and asymmetric RL (_RL (Asym)_), where the policy is still conditioned on the image, but the value function is conditioned on the omniscient compact state [[pin2017a]](#pin2017a), converges slowly, and, are very computationally demanding.  We then show asymmetric imitation learning (_AIL_), and see that this converges almost immediately, but converges to a poor solution.  We start this curve at 80,000 samples, as this is the upfront cost of training the expert, taken to be roughly the number of interactions at which _RL (MDP)_ had converged.  We also compare to using a pre-learned encoder (_Pre-Enc_).  This was learned using rollouts taken under the expert, converting the image into a compact (partial) state encoding.  We see that this is able to recover a good solution in a time commensurate with the _RL (MDP)_, in Frozen Lake, but again requires a number of pretraining examples.  In Tiger Door, this solution is poor as the MDP never pushes the button, and hence the correct required observations are never generated.

We then show our A2D method, when the trainee is provided with a compact-but-partial state encoding (_A2D (Compact)_, i.e. the perfect encoding), and when the trainee is provided with the raw image (_A2D (Image)_).  We see that these methods converge quickly and reliably to the optimal partially observing reward.  Crucially, these converge in a sample budget similar to one another.  This means that the imitation learning is successfully projecting the expert policy onto the high-dimensional trainee policy.  Furthermore, _RL (MDP)_ shows the performance of RL in the MDP, when using comparable hyperparameters to A2D.  This provides an indication of the upper limit on the performance of A2D.  We also find that A2D can function with hyperparameters similar to those tuned for the MDP in isolation.  However, A2D did often benefit from increased entropy regularization, batch size and reduced \\( \lambda \\) values. 


<br>
<center>
<a name="fig3"></a>
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/figures/gridworld/sec4_results_IceLake_True_cr_logs_LavaGap_LavaGapCompiledRun_.png" width="40%"> &nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/figures/gridworld/sec4_results_TigerDoor_True_cr_logs_TigerDoor_TigerDoorCompiledRun_.png" width="40%"> &nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/figures/gridworld/legend_sec4_results.png" width="16.5%">

<small>
_**Figure 3**: Results for the two Gridworld experiments.  **Left**: Frozen Lake.  **Right**: Tiger Door. Results are normalized such that the optimal reward obtainable in the MDP is -1.0 (= -10<sup>0</sup>)._
</small>
</center>
<br>

We also show in [Figure 4](#fig4) the KL-divergence between expert and trainee during learning, for AIL and for A2D with both compact and image-based observations.  We see that the divergence in AIL saturates quickly and to a high divergence, indicating that the trainee is unable to replicate the expert.  We see in A2D that the divergence between the expert and trainee policy is initially low, increases as the expert learns to solve the MDP and as a result of imperfect function approximators and biased gradients, and then reduces to a low divergence (note the log-axis) as the expert is forced in to solve the POMDP in a manner such that the trainee also solves the POMDP, resulting in a low final divergence.  We also see that the convergence paths are commensurate for both compact and image-based representations.  This again suggests that the imitation learning is successfully ameloriating the dimensionality and allowing compact and image-based policies to be learned in with similar dynamics.

<br>
<center>
<a name="fig4"></a>
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/figures/gridworld/sec4_divergence_IceLake_True_cr_logs_LavaGap_LavaGapCompiledRun_.png" width="40%"> &nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/figures/gridworld/sec4_divergence_TigerDoor_True_cr_logs_TigerDoor_TigerDoorCompiledRun_.png" width="40%">

<small>
_**Figure 4**: Divergence results for the two Gridworld experiments.  **Left**: Frozen Lake.  **Right**: Tiger Door.  Note the log-scale on the y-axis._
</small>
</center>
<br>



##### Q-Functions
In the supplement, we also explore the impact that the bias introduced by using the Monte Carlo rollouts, and directly learning the Q-function, has on A2D.  We find that in certain environments, the Q-function must be learned (as opposed to using Monte Carlo rollouts) for A2D to function correctly.  Although this is both a fascinating and challenging quirk, we will not go in to this in detail here, and refer the reader to the supplement.  The key take-home is that to ensure A2D works correctly, the Q-function must be explicitly learned.  However, learning the Q-function increases the computational cost, number of hyperparameters that must be tuned, can introduce bias, and is observed to increase the number of environment interactions required to learn the complete the task.   We also note that while we were able to identify the root cause of this behaviour, we were unable to succinctly define a "test" or "condition" for if a particular environment requires learning a Q-function.  This is an interesting topic for further study.



### Autonomous Vehicles
The second environment we study is an autonomous vehicles (AV) scenario.  We use the AV simulator CARLA [[get ref]](#), and derive this scenario from one of the CARLA Challenge environments [[get ref]](#).  

##### Scenario


##### Results






## <a name="sec_dis"></a> Discussion \& Future Directions
In this work we looked at modifying an asymmetric imitation learning algorithm, such that the expert is modified, or learned online, with the trainee, such that the expert is learned to maximize the reward of the trainee after imitation.  This ameloriates the inherent drawbacks of reinforcement learning in high dimensions, and allows the low-variance supervision provided by AIL to be leveraged.  

In this work we also considered the scenario where the expert is an omniscient policy operating on the complete state representation of the underlying MDP.  An interesting alternative approach is the scenario where, instead of observing a complete state, the expert also only observes a partial state.  That may represent a more generalizable use-case, where expensive and accurate sensors can be used at training time, whereas only cheap, noisy, or incomplete sensor suites are available at test time.  Analysing this scenario is markedly more difficult as one must also reason about information that may be available to the agent, but is not available to the expert.

Beyond this, this work can be viewed as embedding a supervised learning or projection step _inside_ a policy gradient algorithm.  This presents a number of further opportunities for extending our method.  Foremost, one could consider updating the trainee such that the limitations of agent and expert are considered, and vice versa, i.e. the expert and agent are pushed towards policies that provide lower-variance and "better" gradients for each other.  Considering the impact of inexact updates more broadly is a challenging theoretical problem, but that may yield great insights in to the practical aspects of A2D.

Investigating the bias introduced through the Monte Carlo estimator is an immediate and promising avenue of future research.  We present a more thorough discussion of the bias in the supplementary materials.  However, this bias is a product of both the Monte Carlo A2D algorithm _and_ the environment.  Therefore, developing methods for quantifying or bounding this bias, for testing whether the bias is high enough to be terminal for a particular environment and set of hyperparameters, or, developing emperical methods for ameloriating this bias, is potentially the most promising extension.  

Beyond these more concrete points, we believe this general approach is a new and exciting twist and combination of themes from across RL.  Our work is not so much about the particular implementation choices we made, rather that it is possible to pass an importance weighted policy gradient through an imitation learning procedure to update the expert policy, and that is can be cast as embedding a supervised learning task inside an RL task.  

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




## <a name="sec_glo"></a> Glossary & Notation

- **Trainee**: An agent that is learned using imitation learning.
- **Expert**: A policy that is capable of acting optimally in an environment.  
- **Imitation learning** (IL): Learning a policy by regressing onto a second policy (usually an expert).  
- **Compact state**: A complete and low-dimensional representation of the true state of the environment.  
- **Markov decision process** (MDP): See [[war21a, S2.1]](#war21a).  A decision process where the agent/policy can observe the true and complete state of the environment, i.e. there is nothing "unknown".
- **Partially observed Markov decision process** (POMDP): See [[war21a, S2.2]](#war21a).  Extends an MDP to the case where the agent can only observe a partial or noise-corrupted version of the true state.


<br>
<center>
<a name="tab_a1"></a>
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/figures/notation.png" width="100%"> 

<small>
_**Table A.1**: Table of the notation we use throughout the paper and this post._
</small>
</center>
<br>



## <a name="sec_bib"></a> Bibliography

\* Denotes equal contribution.

<a name="war21a"></a> [war21a] **Warrington, A.\*, Lavington, J. W.\*, Ścibior, A., Schmidt, M., & Wood, F. (2021). Robust Asymmetric Learning in POMDPs. _To Appear in International Conference on Machine Learning 2021, arXiv preprint arXiv:2012.15566_.** ([paper](https://arxiv.org/pdf/2012.15566.pdf))

<a name="mur2000a"></a> [mur2000a] Murphy, K. P. (2000). A survey of POMDP solution techniques. _Environment, 2:X3_. ([paper](https://www.cs.ubc.ca/~murphyk/Papers/pomdp.pdf))

<a name="pin2017a"></a> [pin2017a] Pinto, L., Andrychowicz, M., Welinder, P., Zaremba, W., & Abbeel, P. (2017). Asymmetric actor critic for image-based robot learning. _Robotics: Science and Systems XIV_. ([paper](https://arxiv.org/pdf/1710.06542.pdf))

<a name="ros2011a"></a> [ros2011a] Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. _Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (pp. 627-635). JMLR Workshop and Conference Proceedings_. ([paper](http://proceedings.mlr.press/v15/ross11a/ross11a.pdf))

<a name="sch2015a"></a> [sch2015a] Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust region policy optimization. _International conference on machine learning (pp. 1889-1897). PMLR._ ([paper](https://arxiv.org/pdf/1502.05477.pdf))

<a name="wei2020a"></a> [wei2020a] Weihs, L., Jain, U., Salvador, J., Lazebnik, S., Kembhavi, A., & Schwing, A. (2020). Bridging the imitation gap by adaptive insubordination. _arXiv preprint arXiv:2007.12173._ ([paper](https://arxiv.org/pdf/2007.12173.pdf))


<br>
---



© Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, Andrew Warrington & J. Wilder Lavington.

<center>
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/figures/logos/oxford.png" width="20%"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/figures/logos/ubc.png" width="28%"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/figures/logos/iai.png" width="27%"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<embed src="/Users/andrew/Documents/Public_repos/a2d/docs/figures/logos/plai.png" width="9%">
</center>
---
