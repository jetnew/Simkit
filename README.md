# Simkit

A generalized framework for generative and probabilistic modelling to for training reinforcement learning agents in TensorFlow.

<div align="center">
<img src="https://user-images.githubusercontent.com/27071473/88756740-fa213000-d196-11ea-9ce2-d72e935c0b04.png" width="400">
</div>

Many pricing and decision making problems at the core of Grab’s ride-hailing and deliveries business can be formulated as reinforcement learning problems, with interactions of millions of passengers, drivers and merchants from over 65 cities across the Southeast Asia region.

# Usage

```python
from data.synthetic import get_normal_data, plot_data
from model.gmm import GMM

# Get 4 clusters of 1000 normally distributed synthetic data
X, y = get_normal_data(1000, plot=True)

# Fit a Gaussian Mixture Density Network
gmm = GMM(x_features=2,
          y_features=1,
          n_components=32,
          n_hidden=32)
gmm.fit(X, y, epochs=20000)

# Predict y given X
y_hat = gmm.predict(X)
plot_data(X, y_hat)
```

# Models

## Conditional Generative Feature Models

Feature models are used in reinforcement learning for generating features that represent the state during agent-environment interactions.

### Gaussian Mixture Density Network

The Gaussian Mixture Density Network consists of a neural network to predict parameters that define the Gaussian mixture model.

<div align="center">
<img src="https://user-images.githubusercontent.com/27071473/88756971-8c293880-d197-11ea-86a0-a5d658a71d46.png" width="400">
</div>

### Conditional Generative Adversarial Network

The Conditional Generative Adversarial Network consists of a generator network that generates candidate features and a discriminator network that evaluates them, both conditioned on parent features, that contest in optimisation.

<div align="center">
<img src="https://user-images.githubusercontent.com/27071473/88757076-cd214d00-d197-11ea-82fa-110bed7132cd.png" width="400">
</div>

## Probabilistic Response Models

Response models are used in reinforcement learning for the uncertainty modelling of distributional rewards instead of point estimations, to enable stable learning of the agent in cases of spiky responses.

### Bayesian Neural Network

The Bayesian Neural Network is a neural network with weights assigned a probability distribution to estimate uncertainty and trained using variational inference.

<div align="center">
<img src="https://user-images.githubusercontent.com/27071473/88757264-32753e00-d198-11ea-9d89-7868028ec820.png" width="400">
</div>

### Monte Carlo Dropout

The Monte Carlo Dropout is a method shown to approximate Bayesian inference.

<div align="center">
<img src="https://user-images.githubusercontent.com/27071473/88757381-6c464480-d198-11ea-8f81-7734e8bb4894.png" width="400">
</div>


### Deep Ensemble

The Deep Ensemble is an ensemble of randomly-initialised neural networks that performs better than Bayesian neural networks in practice.

<div align="center">
<img src="https://user-images.githubusercontent.com/27071473/88757473-bdeecf00-d198-11ea-9a25-345174caefa7.png" height="200">
</div>

# Utilities

## Performance Metrics

The performance metrics computed are the Kullback-Leibler divergence and Jensen-Shannon divergence, computed by splitting the data into histogram bins.

<div align="center">
<img src="https://user-images.githubusercontent.com/27071473/88758142-55a0ed00-d19a-11ea-90e6-48ee768f6255.png" width="400">
</div>

### Kullback-Leibler Divergence

<div align="center">
<img src="https://latex.codecogs.com/gif.latex?D_{JS}(P||Q)=&space;\frac{1}{2}&space;D_{KL}(P||M)&space;&plus;&space;\frac{1}{2}&space;D_{KL}(Q||M)" title="D_{JS}(P||Q)= \frac{1}{2} D_{KL}(P||M) + \frac{1}{2} D_{KL}(Q||M)" />
</div>

### Jensen-Shannon Divergence

<div align="center">
<img src="https://latex.codecogs.com/gif.latex?M=\frac{1}{2}(P&plus;Q)" title="M=\frac{1}{2}(P+Q)" />
</div>

<div align="center">
<img src="https://latex.codecogs.com/gif.latex?D_{JS}(P||Q)=&space;\frac{1}{2}&space;D_{KL}(P||M)&space;&plus;&space;\frac{1}{2}&space;D_{KL}(Q||M)" title="D_{JS}(P||Q)= \frac{1}{2} D_{KL}(P||M) + \frac{1}{2} D_{KL}(Q||M)" />
</div>
  
## Performance Visualisation

The visualisation tools implemented include the probability density surface plot (left) that visualises the probability densities at each coordinate, and the grid violin relative density plot (right) that visualises the relative densities between the actual data and the generated data of the fitted model using histograms.

<p align="center">
<img src="https://user-images.githubusercontent.com/27071473/88758242-8b45d600-d19a-11ea-917d-25749c3e0a48.png" height="200">
<img src="https://user-images.githubusercontent.com/27071473/88758282-a9133b00-d19a-11ea-8840-e1321ae0253c.png" height="200">
</p>

## Hyperparameter Optimisation using Ax

Hyperparameter optimisation is implemented using Bayesian optimisation in the [Ax](https://ax.dev/) framework, building a smooth surrogate model of outcomes using Gaussian processes from noisy observations from previous rounds of parameterizations to predict performance at unobserved parameterizations, tuning parameters in fewer iterations than grid search or global optimisation techniques.

<div align="center">
<img src="https://user-images.githubusercontent.com/27071473/88758400-ed064000-d19a-11ea-9614-a3279b564777.png" height="200">
</div>

# References
1. Mei, L. I. N., and Christopher William DULA. "Grab taxi: Navigating new frontiers." (2016): 40.
2. Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.
3. Dillon, Joshua V., et al. "Tensorflow distributions." arXiv preprint arXiv:1711.10604 (2017).
4. Bishop, Christopher M. "Mixture density networks." (1994).
5. Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets." arXiv preprint arXiv:1411.1784 (2014).
6. Blundell, Charles, et al. "Weight uncertainty in neural networks." arXiv preprint arXiv:1505.05424 (2015).
7. Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. 2016.
8. Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell. "Simple and scalable predictive uncertainty estimation using deep ensembles." Advances in neural information processing systems. 2017.
9. Fort, Stanislav, Huiyi Hu, and Balaji Lakshminarayanan. "Deep ensembles: A loss landscape perspective." arXiv preprint arXiv:1912.02757 (2019).
10. Chang, Daniel T. "Bayesian Hyperparameter Optimization with BoTorch, GPyTorch and Ax." arXiv preprint arXiv:1912.05686 (2019).
11. Dataset at https://www.kaggle.com/aungpyaeap/supermarket-sales.
12. Dataset at https://www.kaggle.com/binovi/wholesale-customers-data-set.