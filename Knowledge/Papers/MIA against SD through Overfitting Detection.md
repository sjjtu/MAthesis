paper by Breugel, van der Schaar, Sun, Qian , see paper [here](https://arxiv.org/abs/2302.12580)

#mia #genai
## Summary
- New MIA attack Model called DOMIA that infers membership by targetting local overfitting of generative model
- perform MIA under more realistic setting:
	- previous work assume that attacker has no knowledge or access to model params
	- here: attacker has some knowledge on underlying data distribution
- attack model more successful than previous work
- introduce MIA score provides interpretable metrix for privacy

## Motivation
- "*Some generative methods have been shown to memorise samples during the training  procedure, which means the synthetic data samples—which are thought to be genuine—may actually reveal highly private information*" (Carlini et al., 2018) see [[The Secret Sharer- Evaluating and Testing Unintended Memorization in Neural Networks]]
- Problem with DP:
	- either too much privacy and no utility
	- or $\epsilon$ is too big to get any privacy guarantee


## DOMIAS
Intro MIA score also considers true distribution estimated by density estimator
![[Pasted image 20230920150516.png]]
Local overfitting is detected using equation 2: a high score would indicate local overfitting

## Experiment
- use California housing data set
- Use TVAE from [this paper](https://arxiv.org/pdf/1907.00503v2.pdf) which is some kind of GAN + VAE to generate data
- run MIA against the synthetic generated data
## Misc
- good paragraph on mathematical formulation for MIA!
- ![[Pasted image 20230921102532.png]]