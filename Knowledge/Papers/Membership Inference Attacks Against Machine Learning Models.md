#mia #mlaas
## Summary
- MIA in black box setting for classification problems
- propose attack model to distinguish target model's behaviour on training inputs vs non-training inputs -> binary classification problem
- create shadow training technique using generated data
- experimental results indicate high privacy leak when using MLAAS platforms -> median accuracy between 95 and 74 %

## Problem statement
![[Pasted image 20230921123620.png]]
![[Pasted image 20230921123650.png]]

## Generate synthetic data for shadow training
- two steps:
- 1. search using hill climbing algorithm space of possible records and find inputs that are classified with high confidence
- 2. sample from these generated records
- only works if space of possible records is easy to explore, e. g. not high resolution images
- alternative: noisy real data

# Results
- *"By definition, differentially private models limit the success probability of membership
inference attacks based solely on the model, which includes the attacks described in this paper."*
## Misc
- OG paper in MIA in ML, cited over 3000 times
- MIA is not inverse problem see 
- good ref for MLAAS ![[Pasted image 20230921121014.png]]