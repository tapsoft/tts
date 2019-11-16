## **Speaker2Vec**

unsupervised model  
**Goal:** generate speaker embedding without knowing speaker labels

![objective](./img_objective.png)

**Autoencoder architecture**  
![model](./img_model.png)  
input -> K hidden layers -> embedding layer -> K hidden layers -> output  
**basic:** 4000 -> 2000 -> 40 -> 2000 -> 4000  
**modified:** 4000 -> 6000 -> 2000 -> 256 -> 2000 -> 6000 -> 4000

**input:** a small window $$$w_1$$$ of speech (fixed size, $$$d$$$ frames = $$$1.6s$$$, MFCC feature)

**output:** fixed size, $$$d$$$ frames = $$$1s$$$

**training objective:** given $$$w_1$$$, reconstruct a small window $$$w_2$$$ of speech from a temporally neighboring window (fixed size, $$$d$$$ frames = $$$1s$$$, MFCC feature).

**data:** $$$(w_1, w_2)$$$→ a pair. Use all available pairs from an audio input, separated by $$$∆ = 30ms$$$
