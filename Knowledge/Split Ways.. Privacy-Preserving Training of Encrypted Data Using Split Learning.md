# Source: this [paper](https://arxiv.org/pdf/2301.08778.pdf)

In the paper on [Can we use split learning on 1D CNN models for PP training](https://arxiv.org/pdf/2003.12365.pdf) Abuadbba, Kim et al. showed that data can be reconstructed from activation maps of intermediate split layer.
Possible mitigation e. g. adding new hidden layers or using [[Differential Privacy]] leads to less accuracy #tocheck
Authors of this paper propose [[Homomorphic Encryption]] to mitigate privacy leakage


## Analysis
Authors implement different models for comparison:
- **1D CNN model** without encryption as base line
- **U-shaped model** (see [[Split Learning for health]]) first few and last layer on client side, layer(s) in between are on server side, so server does not need label sharing.
	- a plaintext version of this
	- a HE version of this, where the activation maps are encrypted

## Performance
Performance is evaluated on MIT-BIH dataset on heartbeats
- Accuracy not reduced between baseline model and U-shaped model unencrypted
- quantify privacy leakage by measuring correlation of raw input and activation maps by [[distance correlation]] and [[dynamic time warping]].
- HE model is 2-3% worse in accuracy and takes significantly longer in training (three times slower)