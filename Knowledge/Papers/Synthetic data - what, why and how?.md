by James Jordon, Florimond Houssiau, Lukasz Spzruch, Mirko Bottarelli, Giovanni Cherubin, Carsten Maple, Samuel Cohen, Adrian Weller

see [here](https://arxiv.org/abs/2205.03257)

#sd #privacy #intro #genai 
## Summary
- overview on SD with focus on privacy
- vanilla SD is not private see [[Synthetic data- anonymisation groundhog day]]
- SD will always weaken the utility

## Overview on Privacy in ML
- threat model view:
![[Pasted image 20230926120608.png]]
- bayesian view on DP:
![[Pasted image 20230926120748.png]]

## Privacy, Fidelity, Utility
attributes of SD:
### Utility
- usefulness to given task
- dependent on context
- no one-size-fits-it-all solution

### Fidelity
- Treue, Wiedergabetreue, Realitätsnähe
- how does SD statistically matches the OG data

### Privacy
- how much information is revealed about the real data

Generating "good" SD is NP-hard! see


## Evaluating SD
## evaluating privacy
- leakage estimation, see [[fast black-box leakage estimation]]
- simulating privacy attacks: [[Synthetic data- anonymisation groundhog day]]
- 
## Misc
- "Using computer-generated synthetic data to solve particular tasks is not a new
idea, and can be dated back at least as far as the pioneering work of Stanislaw
Ulam and John von Neumann in the 1940s on Monte Carlo simulation methods." page 5
- their definition of SD: 
![[Pasted image 20230926120121.png]]
- SD+DP+FL
![[Pasted image 20230926120441.png]]