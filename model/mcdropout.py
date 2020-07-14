import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Sequential


class MCDropout:
    def __init__(self, x_features, y_features, n_hidden=32, dropout=0.3):
        self.x_features = x_features
        self.y_features = y_features
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.optimizer = tf.keras.optimizers.Adam()
        self.model = self.build()
        
    def model_loss(self, y_true, y_pred):
        """Gaussian negative log likelihood.
            -log p(y|x) = log(sigma^2)/2 + (y-mu)^2/2sigma^2
        Model predits log(sigma^2) rather than sigma^2 for stability."""
        mu = y_pred[:, :self.y_features]
        sigma = y_pred[:, self.y_features:]
        loss = (sigma + tf.square(y_true - mu)/tf.math.exp(sigma)) / 2.0
        return tf.reduce_mean(loss)
    
    def build(self):
        model = Sequential([
            Dense(self.n_hidden, activation='relu'),
            Dense(self.n_hidden, activation='tanh'),
            Dropout(self.dropout),
            Dense(2 * self.y_features, activation=None),
        ])
        model.compile(optimizer=self.optimizer, loss=self.model_loss)
        return model
    
    def loss(self, X, y):
        return self.model_loss(y, self.model.predict(X)).numpy()
    
    def fit(self, X, y, epochs=1000, verbose=1):
        self.model.fit(X, y, batch_size=32, epochs=epochs, verbose=verbose)
        
    def predict(self, X, T=5, return_std=False):
        """Perform T stochastic forward passes."""
        mus, sigmas = [], []
        for t in range(T):
            y_hat = self.model(X, training=True)
            mu, sigma = y_hat[:,:self.y_features], y_hat[:,self.y_features:]
            mus.append(mu)
            sigmas.append(sigma)
        
        mus = np.array(mus)
        sigmas = np.array(sigmas)
        variances = np.exp(sigmas)
        
        y_mean = np.mean(mus, axis=0)
        y_std = np.sqrt(np.mean(variances + mus**2, axis=0) - y_mean**2)
        return y_mean if not return_std else (y_mean, y_std)