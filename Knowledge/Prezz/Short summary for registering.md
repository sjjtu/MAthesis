- [ ] Importance of privacy
	- [x] laws
	- [ ] example breaches
- [ ] Differential privacy
	- [ ] motivation
	- [ ] references
- [ ] Synthetic data
	- [ ] methods
	- [ ] privacy aspect
- [ ] Current state DP + SD?
	- [ ] pitfalls
	- [ ] privacy vs utility
	- [ ] open questions
- [ ] This thesis will answer what
- [ ] How will this thesis answer what



Data-driven technology ans especially machine learning haved gained a lot of momemtum the past years. Models like ChatGPT or BERT heavily depend of large datasets that were available public. At the same time machine learning models are now being considered in other domains like health car **EXAMPLE** or ???. When working with those sensitive data, privacy plays a major role in general acceptance of those models. This is also why governmental institutions like the European Union have established a right to privacy manifested in the General Data Protection Regulation laws. 

That is why one of the reasons why technological advances in the are of privacy-preserving machine learning have increased in the past few years, with the development of various machine learning models that aim to preserve the privacy of of individual data records. One solution might be to replace the original, possibly sensitive data set with a synthetic data set, that resembles the original raw data in some statistical properties. Unfortunately, the naive approach has been shown to still be susceptible to privacy leaks (see ???). To improve the privacy, this thesis aims to combine synthetic data with tools from so-called differential privacy. Differential privacy has been developed by Dwork et al (ref ????) and is widely considered as the mathematical framework to rigorously provide privacy guarantees to privacy-preserving algorithms.