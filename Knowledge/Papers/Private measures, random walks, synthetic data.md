---
Use@: 
Technical topic: 
Overall topic: 
tags: 
authors: 
- March Boedihardjo
- Thomas Strohmer
- Roman Vershynin
source: https://arxiv.org/abs/2204.09167
---


#sd #intro #dp #mp
## Summary

- authors propose a new DP mechanism involving some superregular random walk
- work with metric privacy, more general notion of differential privacy
- results are only valid for Lipschitz statistics
- claim to achieve a polynomial running algo that is accurate for some ML applications
	- prove some bounds

## Private measure
- Given metric space (T,d) we want to to design an algo that transforms a probability measure $\mu$ on T to another probability measure $\nu$ on T
- for empirical measures (discrete data points): 
	- transform original data X into synthethic data Y, s. t. empirical measure are close to each other in Wasserstein 1-metric
	- Wasserstein metric ensures that all lipschitz statistics are "preserved to some extent" using kantorovitch-rubinstein duality

## Background
Metric privacy
![[Pasted image 20230925154425.png]]
which is related to DP via hamming distance metric

## Superregular random walk
- authors construct superregular random walk similar to brownian motion
- but deviation from origin is only logarithmic (vs $\sqrt{n}$  from brownian motion)

## Private measure problem
Find private and accurate algo that transform probability measure to another probability measure on metric space.

Private measure gives private synthetic data:
![[Pasted image 20230925154935.png]]

### Construction:
1. discrete [0,1] interval
2. extend to continuous [0,1] interval via quantization
3. extend to general metric space: map [0,1] to a space filling curve
	- space filling curve can be constructed by using an instance of travelling salesman problem in finite metric space and fold this into the metric space
## Misc
- good ref to human rights: UN General Assembly et al. Universal declaration of human rights. UN General Assembly, 302(2):14–25, 1948.
- ref for companies and US Census using DP, see:
	- Cynthia Dwork, Nitin Kohli, and Deirdre Mulligan. Differential privacy in practice: Expose your epsilons! Journal of Privacy and Confidentiality, 9(2), 2019.
	- John Abowd, Robert Ashmead, Garfinkel Simson, Daniel Kifer, Philip Leclerc, Ashwin Machanavajjhala, and William Sexton. Census topdown: Differentially private data, incremental schemas, and consistency with public knowledge. US Census Bureau, 2019.
	- John M Abowd. The US Census Bureau adopts differential privacy. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2867–2867, 2018