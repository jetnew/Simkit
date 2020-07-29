# Simkit

A generalized framework for generative and probabilistic modelling to for training reinforcement learning agents in TensorFlow.

<img src="https://user-images.githubusercontent.com/27071473/88756740-fa213000-d196-11ea-9ce2-d72e935c0b04.png" width="400">

Many pricing and decision making problems at the core of Grab’s ride-hailing and deliveries business can be formulated as reinforcement learning problems, with interactions of millions of passengers, drivers and merchants from over 65 cities across the Southeast Asia region.

# Models

## Conditional Generative Feature Models

Feature models are used in reinforcement learning for generating features that represent the state during agent-environment interactions.

### Gaussian Mixture Density Network

The Gaussian Mixture Density Network consists of a neural network to predict parameters that define the Gaussian mixture model.

<img src="https://user-images.githubusercontent.com/27071473/88756971-8c293880-d197-11ea-86a0-a5d658a71d46.png" width="400">

### Conditional Generative Adversarial Network

The Conditional Generative Adversarial Network consists of a generator network that generates candidate features and a discriminator network that evaluates them, both conditioned on parent features, that contest in optimisation.

<img src="https://user-images.githubusercontent.com/27071473/88757076-cd214d00-d197-11ea-82fa-110bed7132cd.png" width="400">
  
## Probabilistic Response Models

Response models are used in reinforcement learning for the uncertainty modelling of distributional rewards instead of point estimations, to enable stable learning of the agent in cases of spiky responses.

### Bayesian Neural Network

The Bayesian Neural Network is a neural network with weights assigned a probability distribution to estimate uncertainty and trained using variational inference.

<img src="https://user-images.githubusercontent.com/27071473/88757264-32753e00-d198-11ea-9d89-7868028ec820.png" width="400">

### Monte Carlo Dropout

The Monte Carlo Dropout is a method shown to approximate Bayesian inference.

<img src="https://user-images.githubusercontent.com/27071473/88757381-6c464480-d198-11ea-8f81-7734e8bb4894.png" width="400">


### Deep Ensemble

The Deep Ensemble is an ensemble of randomly-initialised neural networks that performs better than Bayesian neural networks in practice.

<img src="https://user-images.githubusercontent.com/27071473/88757473-bdeecf00-d198-11ea-9a25-345174caefa7.png" height="200">

# Utilities

## Performance Metrics

The performance metrics computed are the Kullback-Leibler divergence and Jensen-Shannon divergence, computed by splitting the data into histogram bins.

<img src="https://user-images.githubusercontent.com/27071473/88758142-55a0ed00-d19a-11ea-90e6-48ee768f6255.png" width="400">

### Kullback-Leibler Divergence

$$ D_{KL}(P||Q)=\sum P(x)log(\frac{P(x)}{Q(x)}) $$

### Jensen-Shannon Divergence

$$ M=\frac{1}{2}(P+Q) $$
$$ D_{JS}(P||Q)= \frac{1}{2} D_{KL}(P||M) + \frac{1}{2} D_{KL}(Q||M) $$
  
## Performance Visualisation

The visualisation tools implemented include the probability density surface plot (left) that visualises the probability densities at each coordinate, and the grid violin relative density plot (right) that visualises the relative densities between the actual data and the generated data of the fitted model using histograms.

<img src="https://user-images.githubusercontent.com/27071473/88758242-8b45d600-d19a-11ea-917d-25749c3e0a48.png" height="200">

<img src="https://user-images.githubusercontent.com/27071473/88758282-a9133b00-d19a-11ea-8840-e1321ae0253c.png" height="200">

## Hyperparameter Optimisation using Ax

Hyperparameter optimisation is implemented using Bayesian optimisation in the [Ax](https://ax.dev/) framework, building a smooth surrogate model of outcomes using Gaussian processes from noisy observations from previous rounds of parameterizations to predict performance at unobserved parameterizations, tuning parameters in fewer iterations than grid search or global optimisation techniques.

<img src="https://user-images.githubusercontent.com/27071473/88758400-ed064000-d19a-11ea-9614-a3279b564777.png" height="200">
