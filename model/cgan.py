import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from mpl_toolkits import mplot3d
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy

class CGAN:
    """Generate y conditioned on x."""
    def __init__(self, x_features, y_features, latent_dim, batch_size=32):
        self.x_features = x_features
        self.y_features = y_features
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.label_smooth = 0.9
        self.g_optimizer = RMSprop(1e-4)
        self.d_optimizer = RMSprop(1e-4)
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.generator.summary()
        self.discriminator.summary()
    
    def build_generator(self):
        noise = Input(shape=(self.latent_dim,))  # noise
        d_noise = Dense(16)(noise)
        x = Input(shape=(self.x_features,))  # condition
        d_x = Dense(16)(x)
        z = Concatenate()([d_noise, d_x])
        d_z = Dense(16)(z)
        y = Dense(self.y_features)(d_z)
        return Model([noise, x], y)
    
    def build_discriminator(self):
        x = Input(shape=(self.x_features))  # condition
        d_x = Dense(16)(x)
        y = Input(shape=(self.y_features))  # y
        d_y = Dense(16)(y)
        h = Concatenate()([d_x, d_y])
        h = Dense(16)(h)
        h = Dropout(0.4)(h)
        p = Dense(1)(h)
        model = Model([y, x], p)
        return model
    
    def g_loss(self, fake_y):
        return -tf.math.reduce_mean(fake_y)
    
    def d_loss(self, real_y, fake_y):
        return -tf.math.reduce_mean(real_y * self.label_smooth) + tf.math.reduce_mean(fake_y)
    
    @tf.function
    def train_step(self, X, real_y):
        noise = tf.random.normal((self.batch_size, self.latent_dim))
        
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_y = self.generator([noise, X], training=True)
            
            real_pred = self.discriminator([real_y, X], training=True)
            fake_pred = self.discriminator([fake_y, X], training=True)
            
            g_loss = self.g_loss(fake_pred)
            d_loss = self.d_loss(real_pred, fake_pred)
            
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Clip discriminator weights
        for var in self.discriminator.trainable_variables:
            var.assign(tf.clip_by_value(var, -0.01, 0.01))
        
        return g_loss, d_loss
    
    def fit(self, dataset, epochs=1000):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/cgan/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        for epoch in range(epochs):
            for X_train, y_train in dataset:
                g_loss, d_loss = self.train_step(X_train, y_train)
            with train_summary_writer.as_default():
                tf.summary.scalar('g_loss', g_loss, step=epoch)
                tf.summary.scalar('d_loss', d_loss, step=epoch)
            if epoch % (epochs // 10) == 0:
                print(f"{epoch} [D loss: {d_loss}] [G loss: {g_loss}]")
        
    def sample(self, X):
        noise = np.random.normal(0, 1, (X.shape[0], self.latent_dim)).astype(np.float32)
        return self.generator([noise, X]).numpy()