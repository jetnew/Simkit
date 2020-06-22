import datetime
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
    def __init__(self, x_features=2, y_features=1, n_components=4, n_hidden=50):
        self.x_features = x_features  # no. of input features
        self.y_features = y_features  # no. of output features
        self.n_components = n_components  # no. of components
        self.n_hidden = n_hidden  # no. of hidden units
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
        
    def tfdGMM(self, pi, mu, sigma):
        """Tensorflow Probability Distributions GMM."""
        batch_size = mu.shape[0]
        mu = tf.reshape(mu, shape=[batch_size, self.n_components, self.y_features])
        sigma = tf.reshape(sigma, shape=[batch_size, self.n_components, self.y_features])
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=pi),
            components_distribution=tfd.MultivariateNormalDiag(loc=mu,
                                                               scale_diag=sigma))
    def loss(self, X, y, training=False):
        pi, mu, sigma = self.model(X, training=training)
        gmm = self.tfdGMM(pi, mu, sigma)
        loss = tf.negative(gmm.log_prob(y))
        return tf.reduce_mean(loss)
    
    @tf.function
    def train_step(self, X, y):
        """TF train function."""
        with tf.GradientTape() as t:
            loss = self.loss(X, y, training=True)
        gradients = t.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
    
    def fit(self, dataset, epochs=1000, plot=False, verbose=True, logdir='gmm'):
        """Fit with TF dataset."""        
        # Tensorboard
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + logdir + '/' + current_time
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        for epoch in range(epochs):
            for train_x, train_y in dataset:
                loss = self.train_step(train_x, train_y)
            with train_summary_writer.as_default():
                tf.summary.scalar('NLL', loss, step=epoch)
            if verbose and epoch % (epochs // 10) == 0:
                print(f"{epoch} [NLL: {loss}]")
        return loss
            
    def prob(self, X, y):
        """Compute probability of y given X."""
        pi, mu, sigma = self.model(X)
        batch_size = mu.shape[0]
        mu = tf.reshape(mu, shape=[batch_size, self.n_components, self.y_features])
        sigma = tf.reshape(sigma, shape=[batch_size, self.n_components, self.y_features])
        y_prob = self.tfdGMM(pi, mu, sigma).prob(y)
        return y_prob 
        
    def sample(self, X):
        """Sample y given X."""
        pi, mu, sigma = self.model(X)
        batch_size = mu.shape[0]
        mu = tf.reshape(mu, shape=[batch_size, self.n_components, self.y_features])
        sigma = tf.reshape(sigma, shape=[batch_size, self.n_components, self.y_features])
        y_pred = self.tfdGMM(pi, mu, sigma).sample()
        return y_pred
    
    def sample_fixed(self, X_fixed, count=20):
        X = np.stack([np.full(count, fill_value=x) for x in X_fixed], axis=1)
        return self.sample(X)