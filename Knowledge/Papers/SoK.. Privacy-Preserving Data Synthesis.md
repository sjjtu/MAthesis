---
Use@:
  - methods
  - sd
Technical topic:
  - sd
Overall topic:
  - survey
tags:
  - survey
  - sd
authors:
  - Yuzheng Hu
  - Fan Wu
  - Qinbin Li
source: https://arxiv.org/pdf/2307.02106.pdf
year: 2023
---


## Summary
- authors give extensive background on state of the art DP synthetic data generators
- test it on MNIST and fashion MNIST and then train different classifiers with the synthetic data
- DP-MERF is best overall according to authors
## Background
- 

## Misc
- evaluation metric for fidelity:
	- frechet inception distance
	- inception score
- utility:
	- classification accuracy
- utility and fidelity are correlated here
- *"The pessimistic conclusion in Stadler et al. does not
fully extend to image data."* referring to [[Synthetic data- anonymisation groundhog day]]
