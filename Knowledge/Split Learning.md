Mentioned in this [article](https://www.media.mit.edu/projects/distributed-learning-and-collaborative-learning-1/overview/) by MIT media lab, based on this paper on [[Papers/Split Learning for health]]

Part of this paper on [[Papers/Split Ways.. Privacy-Preserving Training of Encrypted Data Using Split Learning]]

## Basic idea
Each client trains a partial deep neural network up to a specified layer - the cut layer. Outputs are sent to another party (server / another client) which continues training from cut layer, thus never receiving raw data.
For backpropagation, the other way around: gradients are computed from last layer to cut layer and then send to another entity, where the gradients are further backpropagated.

How will inference work? Will the client have the whole model weights? 
-> probably just like a forward pass

How are the local weights influenced by other clients's data? #todo 