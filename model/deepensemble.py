import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Sequential


class DeepEnsemble:
    def __init__(self, x_features, y_features,
                 n_models=5,
                 n_hidden=32,
                 dropout=0.3):
        self.x_features = x_features
        self.y_features = y_features
        self.n_models = n_models
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.models = [self.build() for _ in range(self.n_models)]
        
    def model_loss(self, y_true, y_pred):
        """Gaussian negative log likelihood.
            -log p(y|x) = log(sigma^2)/2 + (y-mu)^2/2sigma^2
        Model predits log(sigma^2) rather than sigma^2 for stability."""
        mu, sigma = y_pred[:,:self.y_features], y_pred[:,self.y_features:]
        loss = (sigma + tf.square(y_true - mu)/tf.math.exp(sigma)) / 2.0
        return tf.reduce_mean(loss)
    
    def build(self):
        model = Sequential([
            Dense(self.n_hidden, activation='relu'),
            Dense(self.n_hidden, activation='tanh'),
            Dropout(self.dropout),
            Dense(2 * self.y_features)])
        model.compile(loss=self.model_loss, optimizer=self.optimizer)
        return model
    
    def loss(self, X, y):
        return sum([self.model_loss(y, model.predict(X))
                    for model in self.models]) / self.n_models
    
    def fit(self, X, y, epochs=1000, verbose=1):
        """Fit ensemble models by random initialisation."""
        for i in range(self.n_models):
            self.models[i].fit(X, y, batch_size=32, epochs=epochs, verbose=verbose)
        
    def predict(self, X, return_std=False):
        """Perform T stochastic forward passes."""
        mus, sigmas = [], []
        for model in self.models:
            y_hat = model.predict(X)
            mu, sigma = y_hat[:,:self.y_features], y_hat[:,self.y_features:]
            mus.append(mu)
            sigmas.append(sigma)
            
        mus = np.array(mus)
        sigmas = np.array(sigmas)
        variances = np.exp(sigmas)
        
        y_mean = np.mean(mus, axis=0)
        y_variance = np.mean(variances + mus**2, axis=0) - y_mean**2
        y_std = np.sqrt(y_variance)
        return y_mean if not return_std else (y_mean,  y_std)