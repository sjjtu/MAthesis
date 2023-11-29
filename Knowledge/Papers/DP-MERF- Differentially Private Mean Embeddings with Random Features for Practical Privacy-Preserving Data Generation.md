---
Use@:
  - sd
  - sota
Technical topic:
  - sd
Overall topic:
  - sd
tags:
  - privacy
  - sd
  - dp
authors:
  - Frederik Harder
  - Kamil Adamczewski
  - Mijung Park
source: https://arxiv.org/pdf/2002.11603.pdf
year: 2021
---


## Summary
- Goal: train generator $G_{\theta}(z)$  that is parametrised by $\theta$ and takes as input some random vector $z \sim p(z)$ from some known data-independent distribution $p$.
- train generator via learning the distribution by minimising "random feature representation" of MMD: $$\hat{\theta}=argmin_{\theta} \tilde{MMD}^2(P_{x,}Q_{x_\theta})$$ where $P$ is the underlying true data distribution and $x_\theta$ is drawn from generator
- using random fourier features, we can rewrite the MMD using "kernel mean embeddings"$$\tilde{MMD}^2(P,Q)=||\hat{\mu}_{P}-\hat{\mu}_{Q}||^2$$
- since $\mu_p$ is only data-dependent term, we add noise to this part $$\mu_{p}= \mu_{p}+ gausssian$$ 
## Background
- 

## Misc
- github: https://github.com/ParkLabML/DP-MERF/tree/master/code_tab