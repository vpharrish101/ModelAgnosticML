### Meta Learners

## What are they?

Meta learners are basically a new way to train a NN model - be it a graph, CNN, etc.
  1. Generally, we train a model per epoch by making it adapt to the entire dataset. Say if epoch=5 and data=20, we make the model learn from 20 points, 5 times
  2. In this, the dataset itself is randomly sampled into a {S:Q} subset, and used to train on.
  3. First, we clone the model's params. Then, we take 20 of these {S,Q} datasets, train the clone on it.
  4. Now, we evaluate the clone with a loss fn, do it a couple times and those loss graphs are used to train the original model.

## Why do it that way?
Simple.
  1. It makes the model robustly resistant to overfitting
  2. Model can learn underlying patterns even with fewer datasets
  3. "On-the-fly" training is possible, where you can introduce a new class into the testing dataset and it will learn to adapt for those too.


## Code details: -

MAML: Model-Agnostic Meta Learning (standard)
protoMAML: Prototypical Networks - based MAML. 
OptNet: Solves a QP model, and estimating the Gaussian RBF matrix via Random Fourier Features, massively reducing time and compute.
