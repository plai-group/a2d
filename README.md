<!-- # Source Code for "Robust Asymmetric Learning in POMDPs" !-->

# Adaptive Asymmetric DAgger (A2D)

This repository contains the source code accompanying the paper Robust Asymmetric Learning in POMDPs, Warrington, A.\*, Lavington, J. W.\*, Ścibior, A., Schmidt, M., & Wood, F. (2021). Robust Asymmetric Learning in POMDPs. _International Conference on Machine Learning, 2021.  arXiv preprint arXiv:2012.15566_. ([paper](https://arxiv.org/pdf/2012.15566.pdf)).


## A Foreword: Multiple Repositories

While working remotely and in different timezones as a result of COVID-19, the long route to market for this paper, and one of the authors continuing to work on the project after the other had finished the degree program, the codebases used for the two main experiments in the paper (gridworld and autonomous vehicles (AV)) diverged.  

Although the core of each algorithm is the same, there are several differences in the low-level implementation that make combining the codebases for exact reproduction of experimental results (tied to the exact Git commit or Weights and Biases experimental log) (a) an onerous task, and (b) would make the code unnecessarily complex and hard to parse.  

While it is on the to-do list to reconcile these codebases, the most sensible thing to do in the immediate future is to simply release both codebases verbatim.  This ensures that the results in the paper are immediately reproducible, and the _exact_ code used to generate those results can be inspected, critiqued, and built upon.  Each codebase can also stripped back for specifically that individual experiment, making the code as readily understandable as possible.  Finally, proposing bugfixes etc to each codebase individually reduces the chance that it _silently_ breaks the ability of the other codebase to reproduce results.  

In the future, once a unified codebase is created, it will be inserted directly into this repository.  The original repositories will then be linked as submodules (or similar) for indefinite preservation of experimental reproducibility and provenance.  



## Core Installation

To download and install this repository run:

```
git clone https://github.com/plai-group/a2d
cd a2d
pip install -r requirements.txt
```

## Layout

This repository contains three things: the gridwold implementation; the AV implementation; and the accompanying materials.

The gridworld and AV implementations are essentially standalone codebases contained within this "root" repository.  The root repository contains the materials the accompanying materials, including the LaTeX source code for the paper, figures, [poster](https://github.com/plai-group/a2d/blob/master/docs/poster/a2d_poster_v2.pptx), materials used in the [talk](https://github.com/plai-group/a2d/tree/master/docs/talk), and an informal [blog post](https://github.com/plai-group/a2d/tree/master/docs/blog).  Each implementation is individually documented within the respective directory. 

The gridworld implementation is contained in `a2d_gridworld`.  The gridworld implementation requires no further installation beyond installing the `requirements.txt` through pip.

The AV implementation is contained in `a2d_av`.   The AV implementation requires a working installation of the AV simulator CARLA.  Information about installing the required software and containers is contained in `a2d_av`. 





## Contact

If you have questions, comments, bugfixes, or ideas for collaborations and extensions, then we are all ears!  Please reach out to us at `awarring at stanford dot edu` and we would be more than happy to help out wherever we can! :) 

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



