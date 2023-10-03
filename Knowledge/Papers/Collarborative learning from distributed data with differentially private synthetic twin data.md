---
Use@:
  - intro
  - examples
  - dataset
Technical topic:
  - distributedlearning
  - sd
Overall topic:
  - health
tags:
  - sd
  - distributedlearning
authors:
  - Lukas Prediger
  - Joonas Jälkö
  - Antti Honkela
  - Samuel Kaski
source: https://arxiv.org/abs/2308.04755
---


## Summary
- setting where multiple parties hold sensitive data aim to collaboratively learn a population level statistics, but remain private
- setting is health care data from UK biobank
- authors get a better result than just using local data
## Background
- assume more than 2 parties that want to perform statistical analyses over population
- each party $m$ has access to local data $D_m$ that disjoint and non-uniformly sampled from overall distrbution, i. e. $P(x|m) \neq P(x)$ 
- each party trains generative model on local data and published synthetic data in place of sensitive
- formal requirements:
	- disjoint local maybe skewed data
	- parties' combined data represent overall population
	- each party's goal is optimise their analysis of population and do not actively try to negatively affect other parties (non-malicious clients)
	- results of each parties's analysis are kept private
	- party receiving synthetic will also publish own synthetic data
- Question / dilemma: "*apparent dilemma: If the local data of a party m is not sufficient
to learn the analysis model for the global population well, this suggests that it might not be
possible to learn a good generative model from it either, especially under privacy constraint. [...] Does incorporating (low-fidelity) synthetic data generated from small data sets of other parties improve results of the analysis performed by party m over just using its own (small) local data set? *" (page 3)
-> answer: yes!

## Method
- use UK Biobank on risk of individual to test positive for covid based on socio-economic factors
- data set naturally splits into 16 assessment centers by geographic location -> 16 clients
- real world: each client does not hold a lot of data -> authors use a subsample of 10% of training data and increase it later
- different parameters:
	- how much synthetic data is needed? -> fix a center and successively add center and evaluate log-likelihood
	- effect of local data size -> quality of SD decreases more quickly than quality of regression model trained only on local data depending on number of data points?
	- is data sharing beneficial for parties with large data set? -> isolate largest assessment center (Newcastle)

## Results
### number of synthetic data sets:
	![[Pasted image 20231003111408.png]]
	- log-likelihood increases
	- spread also increases initially
	- after adding 5 data sets increase gets smaller

### size of local data
- subsample 10, 20, 50 and 100 % of training data
![[Pasted image 20231003112105.png]]

*"Even when the individual sets are small and of poor quality,
in combination they still carry an overall strong enough signal to enable meaningful analysis.
The results for other centers are consistent with this."* (page 6)
## Misc
- UK biobank data set can be used as data