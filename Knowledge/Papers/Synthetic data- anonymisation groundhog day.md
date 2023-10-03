---
Use@:
  - sota
  - methods
Technical topic:
  - sd
  - privacy
Overall topic:
  - review
  - framework
tags:
  - mia
  - sd
  - genai
  - sanitisation
authors:
  - Theresa Stadler
  - Bristena Oprisanu
  - Carmela Troncoso
source: https://arxiv.org/abs/2011.07018
---

## Summary
- open source framework for evaluating privacy gain of SD and compare to anonymisation techniques
- empirically shows that SD does not retain data utility or privacy
- SD leads to variable privacy gain and unpredictable utility loss
- misconception about SD: previous studies underestimate risk of SD
- Attack model: SD gen is black box, focus on tabular data

### Synthetic data
good description of SD
![[Pasted image 20230922114759.png]]

## Experiment
- 5 different models:
	- IndHist: extract marginal frequency counts, continuous attributes are binned
	- BayNet: bayesian network capturing correlations between attributes
	- PrivBayes: DP bayesian network
	- CTGAN: bases on GAN plus some improvements
	- PATEGAN: DP for GANs
- evaluate two groups of targets: random targets and outliers

### Evaluation framework
measure privacy gain by difference between adversary's advantage when given synthetic data S instead of raw data R
![[Pasted image 20230922115555.png]]

### Formalising MIA as linkability attacks
- linkability game:
	- Challenger: data holder
	- adversary: infer whether a target record $r_{t}$ is present it sensitive raw data R based on published data X and prior knowledge P
		![[Pasted image 20230922120708.png]]
### Results:
- for gen ai models: privacy gain is hard to predict
- for DP models: provide higher privacy gain, but need to carefully check theoretical assumptions in order to achieve theoretical results, but comes at cost of utility
- ![[Pasted image 20231003134157.png]]