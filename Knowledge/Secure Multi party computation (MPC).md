Given participants $p_i$ and private date $d_i$ for $i=1...,N$ we want to compute a global function $F(d_1,...,d_N)$ without revealing the private data.

*Example*: 
 - Alica, Bob and Charlie
 - x,y,z denote their salary which is only known by the person
 - Want to find out the highest salary, i. e. $F(x,y,z)=max(x,y,z)$
 - Optimal scenario: there is a trusted outside party Tony, that is trusted by everyone and they can send Tony the data, Tony computes the maximum and returns the maximum number to everyone
 - **We want to find a protocol without relying on third party!**

Informally, basic properties of SMC:
- Input privacy: no information about private data can be inferred from messages during execution of protocol
- Correctness: honest parties are either guaranteed to compute correct result ([[Robust]] protocol)  or they abort

