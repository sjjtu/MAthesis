---
Use@:
  - dp
  - methods
Technical topic:
  - privacy
  - dp
  - dpsgd
Overall topic:
  - dp
tags:
  - momentsaccountant
  - dpsgd
authors:
  - Ian Goodfellow
  - Martin Abadi
  - Andy Chu
  - Kunal Talwar
  - Ilya Mironov
  - Li Zhang
source: https://arxiv.org/pdf/1607.00133.pdf
year: 2016
---


## Summary
- Authors introduce differentially private SGD
- track privacy loss using moments of privacy loss
- test on MNIST and CIFAR
## Background
- summary of DP properties:
	- Composability: enable modular design of mechanisms
	- Group privacy: graceful degradation of privacy guarantees if dataset contains correlated input, e. g. records from same person
	- Robustness to auxiliary information: privacy guarantees are not affected by any side information

## DPSGD
steps:
	1. compute gradient for random subset of examples
	2. clip $l_2$ norm of gradient and average
	3. add noise to protect privacy
	4. keep track of privacy loss

## Moments accountant
## Misc
- example of model-inversion attack
	- M. Fredrikson, S. Jha, and T. Ristenpart. Model inversion attacks that exploit confidence information and basic countermeasures. In CCS, pages 1322–1333. ACM, 2015