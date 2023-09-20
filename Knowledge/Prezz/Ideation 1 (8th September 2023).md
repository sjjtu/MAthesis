---

---
## Ideation    - 8th Sept 2023
### Agenda
1. Motivation to Synthetic Data
2. Synthetic Data
3. Differential Privacy
4. DP + Synthetic Data?
  
---
## Motivation to Synthetic Data

- [ MIT Tech' Review](https://www.technologyreview.com/2022/02/23/1045416/10-breakthrough-technologies-2022/#synthetic-data-for-ai) (2022): *" ... Breakthrough Technology ... "*
 - [Gartner](https://www.gartner.com/en/newsroom/press-releases/2022-06-22-is-synthetic-data-the-future-of-ai) (2022): *" ... estimates that by 2030, synthetic data will completely overshadow real data in AI models"*

--- 

![[Images/Pasted image 20230907131418.png  ]]

---
## Synthetic Data (SD)

**Informal Definition** (Synthetic Data) taken from [lecture notes](https://tamids.tamu.edu/wp-content/uploads/2021/10/Slides-Thomas-Strohmer.pdf) 
Synthetic data is generated from existing data, such that statistical properties of the original data is maintained, but without risk of exposing sensitive information.

-> Tradeoff: privacy vs utility
	Extreme cases:
		Synthetic data = original data: Perfect utility, zero privacy
		Synthetic data = random data: Perfect privacy, zero utility.
		
---
## Differential Privacy (DP)

OG definition by Dwork et al (2006)

**Definition** (Differential Privacy) 
A mechanism $\mathcal{M}$ gives $\epsilon$ - differential privacy if for all data sets $D$ and $D'$ that only differ in one element and all subsets $\mathcal{S} \in \text{ran } \mathcal{M}$   :
$$ \mathbb{P}(\mathcal{M}(D) \in \mathcal{S}) \le \exp(\epsilon) \cdot \mathbb{P}(\mathcal{M}(D') \in \mathcal{S}) $$
for small $\epsilon << 1$ we have the relation $\exp(\epsilon) \approx 1+ \epsilon$ , which gives
$$  \mathbb{P}(\mathcal{M}(D) \in \mathcal{S}) \le (1+\epsilon) \cdot \mathbb{P}(\mathcal{M}(D') \in \mathcal{S})  $$

*In plain words: inclusion or exclusion of one data entry should not change the out come "much". "Much" is measured by the privacy loss $\epsilon$.*


---
### How big should $\epsilon$ be?

[Review](https://arxiv.org/pdf/2206.04621.pdf) on different $\epsilon$ in practice:

| epsilon | used by |
|--| -|
| $\le$ 1 | recommended by OG Dwork|
| 0.5 - 8 | highly cited literature |
| 9 | Google |
| 6-43 | Apple |


---
### How to achieve DP "traditional"?

Add noise according to:
- laplacian distribution
- exponential distribution
- geometric disitribution

---
## DP + SD = ???
### Approach 1: Gen AI
- [PATE-GAN](https://openreview.net/pdf?id=S1zk9iRqF7) (2019) by Jordon, Yoon, van der Schaar: 
	- uses relaxed version of DP, but achieves state of the art results to generate synthetic data
	- very good $\epsilon=1$

---
## DP + SD = ???
### Approach 2: Noise
- interesting [paper](https://www.math.ucdavis.edu/~strohmer/papers/2022/privatemeasure.pdf) (2022) by Boedihardjo, Strohmer, Vershynin: 
	- very mathematical approach
	- construct efficient private synthetic data that is accurate in many statistical operations
	- relies on more general notion of DP -> metric DP
	- maybe leverage this technique ????
---

## Brainstorming


---
