## **Speaker2Vec**

unsupervised model  
**Goal:** generate speaker embedding without knowing speaker labels

![objective](./img_objective.png){:width="200"}

**Autoencoder architecture**  
![model](./img_model.png){:width="200"}
input -> K hidden layers -> embedding layer -> K hidden layers -> output  
**basic:** 4000 -> 2000 -> 40 -> 2000 -> 4000  
**modified:** 4000 -> 6000 -> 2000 -> 256 -> 2000 -> 6000 -> 4000

**input:** a small window w1 of speech (fixed size, d frames = 1.6s MFCC feature)

**output:** fixed size, d frames = 1s

**training objective:** given w1 reconstruct a small window w2 of speech from a temporally neighboring window (fixed size, d frames = 1s MFCC feature).

**data:** (w1, w2) -> a pair. Use all available pairs from an audio input, separated by ∆ = 30ms
