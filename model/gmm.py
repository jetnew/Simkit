import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from mpl_toolkits import mplot3d
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class GMM:
    """Gaussian Mixture Density Network."""
    def __init__(self, x_features=2, y_features=1, n_components=4, n_hidden=50, verbose=False):
        self.x_features = x_features  # no. of input features
        self.y_features = y_features  # no. of output features
        self.n_components = n_components  # no. of components
        self.n_hidden = n_hidden  # no. of hidden units 
        self.verbose = verbose
        self.build()
    
    def build(self):
        """Compile TF model."""
        input = tf.keras.Input(shape=(self.x_features,))
        layer = tf.keras.layers.Dense(self.n_hidden, activation='tanh')(input)

        mu = tf.keras.layers.Dense(self.n_components * self.y_features)(layer)
        sigma = tf.keras.layers.Dense(self.n_components * self.y_features, activation='exponential')(layer)
        pi = tf.keras.layers.Dense(self.n_components, activation='softmax')(layer)

        self.model = tf.keras.models.Model(input, [pi, mu, sigma])
        self.optimizer = tf.keras.optimizers.Adam()
        
        if self.verbose:
            print(self.model.summary())
        
    def tfdGMM(self, pi, mu, sigma):
        """Tensorflow Probability Distributions GMM."""
        mu = tf.reshape(mu, (self.n_components, self.y_features))
        sigma = tf.reshape(sigma, (self.n_components, self.y_features))
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=pi),
            components_distribution=tfd.MultivariateNormalDiag(loc=mu,
                                                               scale_diag=sigma))

    def loss(self, y, pi, mu, sigma):
        """Loss function, negative log-likelihood."""
        samples = pi.shape[0]
        losses = 0
        for i in range(samples):
            gmm = self.tfdGMM(pi[i], mu[i], sigma[i])
            loss = gmm.log_prob(y[i])
            loss = tf.negative(loss)
            losses += loss
        return losses / samples
    
    @tf.function
    def train_step(self, X, y):
        """TF train function."""
        with tf.GradientTape() as t:
            pi, mu, sigma = self.model(X, training=True)
            loss = self.loss(y, pi, mu, sigma)
        gradients = t.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
    
    def fit(self, dataset, epochs=1000, plot=False):
        """Fit with TF dataset."""
        losses = []
        print_every = int(0.1 * epochs)
        
        for i in range(epochs):
            for train_x, train_y in dataset:
                loss = self.train_step(train_x, train_y)
            losses.append(loss)
            if self.verbose and i % print_every == 0:
                print('Epoch {}/{}: Negative Log-Likelihood {}'.format(i, epochs, losses[-1]))
        
        if plot:
            plt.plot(range(len(losses)), losses)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Negative Log-Likelihood')
            plt.show()
            
    def prob(self, X, y):
        """Compute probability of y given X."""
        pi, mu, sigma = self.model(X) 
        samples = pi.shape[0]
        y_prob = []
        for i in range(samples):
            y_prob.append(self.tfdGMM(pi[i], mu[i], sigma[i]).prob(y[i]).numpy())
        return np.array(y_prob)
        
    def sample(self, X):
        """Sample y given X."""
        pi, mu, sigma = self.model(X) 
        samples = pi.shape[0]
        y_pred = []
        for i in range(samples):
            y_pred.append(self.tfdGMM(pi[i], mu[i], sigma[i]).sample().numpy())
        return np.array(y_pred)
    
    def sample_fixed(self, X_fixed, count=20):
        X = np.stack([np.full(count, fill_value=x) for x in X_fixed], axis=1)
        return self.sample(X)