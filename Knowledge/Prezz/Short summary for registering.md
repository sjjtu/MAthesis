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



Data-driven technology and especially machine learning have gained a lot of momentum the past years. Models like ChatGPT or BERT heavily depend on large datasets that were available publicly. At the same time machine learning models are now being considered in other domains like health care **EXAMPLE** or ???. 

When working with those sensitive data, privacy plays a major role in general acceptance of those models. This is also why governmental institutions like the European Union have established a right to privacy manifested in the General Data Protection Regulation laws. Previous simple anonymisation attempts that just removed some identifying attributes have been proven to be ineffective. For example, user profiles from the anonymised dataset used in the infamous Netflix price (REF??) have been reconstructed with the help of publicly available data from IMDB.

That is one of the reasons why technological advances in the area of privacy-preserving machine learning have increased in the past few years, with the development of various machine learning models that aim to preserve the privacy of individual data records. One promising solution (REF???) might be to replace the original, possibly sensitive data set with a synthetic data set that resembles the original raw data in some statistical properties. Unfortunately, the naive approach has been shown to still be susceptible to privacy leaks (see ???). To improve privacy, this thesis aims to analyse the combination of synthetic data with tools from so-called differential privacy. Differential privacy has been developed by Dwork et al (ref ????) and is widely considered as the mathematical framework to rigorously provide privacy guarantees to privacy-preserving algorithms. This thesis will study existing architecture based on private generative AI models, as well as explore the possibility of new solution. To assess the theoretical findings empirically, further practical attacks, e. g. membership inference attacks are performed. Unfortunately, there is no free lunch and privacy always comes with a decrease in utility (REF???). A careful balance between privacy and utility needs to be established.