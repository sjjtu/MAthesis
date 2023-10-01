---
Use@: 
Technical topic: 
Overall topic: 
- health
tags: splitLearning
authors: 
source: https://arxiv.org/pdf/2003.12365.pdf
---

### Summary
Authors design a 1D CNN for sequential heartbeat data to detect heart rhythm diseases. They use [[Split Learning]] and found some privacy issues that they try to mitigate using i) more hidden layers and ii) apply a [[Differential Privacy]] mechanism to add noise to data.

### Method
Compare two different architectures:
- two layer CNN with two FC layers
- three layer CNN with two FC layers
Splitting after CNN layer

Privacy measures:
- visual invertibility:
	- visualize each channel in raw input and output from last CNN layer
- distance correlation
	- measure "distance" between random variables
- Dynamic time warping
	- similarity measurement for time series analysis
	- measure similarity between two temporal sequences which may vary in speed
### Mitigation measures
**Adding more hidden layers**
- adding 2 to 8 hidden layers
- only slight improvement in privacy measures
- accuracy drops by around 5%
- but introduces a lot of computational overhead which defies to reason to use Split learning to begin with

**Differential Privacy**
- use laplace differential privacy mechanism
- lest out different noise levels from 10 (weakest privacy) to 0 (strongest privacy but data cannot be used)
- accuracy drops by 40 % in worst case


### Interesting
- authors claim that split learning with LSTM or RNN is not yet investigated -> maybe something for [[Ideas]]