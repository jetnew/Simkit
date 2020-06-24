import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy

class CGAN:
    """Generate y conditioned on x."""
    def __init__(self, x_features, y_features, latent_dim=32, g_hidden=16, d_hidden=16, label_smooth=0.9, d_dropout=0.2, d_clip=0.01):
        self.x_features = x_features
        self.y_features = y_features
        self.latent_dim = latent_dim
        self.g_hidden = g_hidden
        self.d_hidden = d_hidden
        self.label_smooth = label_smooth
        self.d_dropout = d_dropout
        self.d_clip = d_clip
        self.g_optimizer = RMSprop(1e-4)
        self.d_optimizer = RMSprop(1e-4)
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
    
    def build_generator(self):
        """Generator model consists of a dense layer after each component."""
        noise = Input(shape=(self.latent_dim,))  # noise
        d_noise = Dense(self.g_hidden)(noise)
        d_noise = Dense(self.g_hidden)(d_noise)
        x = Input(shape=(self.x_features,))  # condition
        d_x = Dense(self.g_hidden)(x)
        d_x = Dense(self.g_hidden)(d_x)
        z = Concatenate()([d_noise, d_x])
        d_z = Dense(self.g_hidden)(z)
        d_z = Dense(self.g_hidden)(d_z)
        y = Dense(self.y_features)(d_z)
        return Model([noise, x], y)
    
    def build_discriminator(self):
        """Discriminator model consists of a dense layer after each component."""
        x = Input(shape=(self.x_features))  # condition
        d_x = Dense(self.d_hidden)(x)
        d_x = Dense(self.d_hidden)(d_x)
        y = Input(shape=(self.y_features))  # y
        d_y = Dense(self.d_hidden)(y)
        d_y = Dense(self.d_hidden)(d_y)
        h = Concatenate()([d_x, d_y])
        h = Dense(self.d_hidden)(h)
        h = Dense(self.d_hidden)(h)
        h = Dropout(self.d_dropout)(h)
        p = Dense(1)(h)
        return Model([y, x], p)
    
    def g_loss(self, fake_y):
        return -tf.math.reduce_mean(fake_y)
    
    def d_loss(self, real_y, fake_y):
        return -tf.math.reduce_mean(real_y * self.label_smooth) + tf.math.reduce_mean(fake_y)
    
    def gradient_penalty(self, real_y, fake_y, X):
        """Gradient penalty on discriminator"""
        batch_size = real_y.shape[0]
        alpha = tf.random.normal([batch_size, self.y_features], 0.0, 1.0)
        diff = fake_y - real_y
        interpolated = real_y + alpha * diff
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator([interpolated, X], training=True)
            
        gradients = gp_tape.gradient(pred, [interpolated])
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def diversity_score(self, X):
        batch_size = X.shape[0]
        z1 = tf.random.normal([batch_size, self.latent_dim])
        z2 = tf.random.normal([batch_size, self.latent_dim])
        y1 = self.generator([z1, X], training=True)
        y2 = self.generator([z2, X], training=True)
        denom = tf.reduce_mean(tf.abs(z1 - z2), axis=1)
        numer = tf.reduce_mean(tf.abs(y1 - y2), axis=1)
        ds = tf.reduce_mean(numer/denom)
        t = 0.1  # lower bound for numerical stability
        return tf.math.minimum(ds, t)   
    
    @tf.function
    def train_step(self, X, real_y):
        noise = tf.random.normal((X.shape[0], self.latent_dim))
        
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_y = self.generator([noise, X], training=True)
            
            real_pred = self.discriminator([real_y, X], training=True)
            fake_pred = self.discriminator([fake_y, X], training=True)
            
#             gp = self.gradient_penalty(real_y, fake_y, X)
            ds = self.diversity_score(X)
            
            g_loss = self.g_loss(fake_pred) - ds
            d_loss = self.d_loss(real_pred, fake_pred)# + gp
            
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Clip discriminator weights
        for var in self.discriminator.trainable_variables:
            var.assign(tf.clip_by_value(var, -self.d_clip, self.d_clip))
        
        return g_loss, d_loss
    
    def fit(self, dataset, epochs=300, verbose=True, logdir='cgan'):
        # Tensorboard
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + logdir + '/' + current_time
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        for epoch in range(epochs):
            for X_train, y_train in dataset:
                g_loss, d_loss = self.train_step(X_train, y_train)
            with train_summary_writer.as_default():
                tf.summary.scalar('Generator Loss', g_loss, step=epoch)
                tf.summary.scalar('Discriminator Loss', d_loss, step=epoch)
            if verbose and epoch % (epochs // 10) == 0:
                print(f"{epoch} [D loss: {d_loss}] [G loss: {g_loss}]")
        
    def sample(self, X):
        noise = tf.random.normal((X.shape[0], self.latent_dim))
        return self.generator([noise, X]).numpy()