import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class BayesNN:
    """Bayesian neural network using Flipout estimator"""
    def __init__(self, x_features, y_features, n_hidden=32, n_layers=3):
        self.x_features = x_features
        self.y_features = y_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.optimizer = tf.keras.optimizers.Adam()
        self.model_loss = lambda y, distr: -distr.log_prob(y)
        
    def kernel_divergence_fn(self, n_samples):
        return lambda q, p, _: tfp.distributions.kl_divergence(q, p) / n_samples
    
    def bias_divergence_fn(self, n_samples):
        return lambda q, p, _: tfp.distributions.kl_divergence(q, p) / n_samples
        
    def build(self, n_samples):
        model = tf.keras.Sequential([])
        for n in range(self.n_layers + 1):
            model.add(
                tfp.layers.DenseFlipout(self.n_hidden if n < self.n_layers else 2*self.y_features,
                    bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                    bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                    kernel_divergence_fn=self.kernel_divergence_fn(n_samples),
                    bias_divergence_fn=self.bias_divergence_fn(n_samples),
                    activation="relu"))
        model.add(tfp.layers.DistributionLambda(
            lambda params: tfd.MultivariateNormalDiag(
                loc=params[:,:self.y_features],
                scale_diag=1e-3+tf.math.softplus(0.05*params[:,self.y_features:]))))
        model.compile(self.optimizer, self.model_loss)
        return model
    
    def fit(self, X, y, epochs=1000, batch_size=32, verbose=1):
        self.model = self.build(n_samples=X.shape[0])
        self.model.fit(X, y,
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=verbose)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def loss(self, X, y):
        return tf.reduce_mean(self.model_loss(y, self.model(X)))