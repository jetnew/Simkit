import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from mpl_toolkits import mplot3d
import tensorflow as tf


def plot_data(X, y):
    """Plot 2D or 3D according to X dimension."""
    if X.shape[1] == 1:
        for i in range(y.shape[1]):
            plt.plot(X, y[:,i], '.')
            plt.show()
    if X.shape[1] == 2:
        for i in range(y.shape[1]):
            ax = plt.axes(projection='3d')
            ax.scatter(X[:,0], X[:,1], y[:,i], c=y[:,i], cmap='viridis')
            plt.show()
            

def get_normal_data(n=1000, plot=False, xy_features=(2,1)):
    """Get normal dataset of x=(n*4,2), y=(n*4,1).
       Synthetic data is normally distributed at 4 clusters:
           X = (20,60), y = (100)
           X = (20,80), y = (120)
           X = (40,60), y = (140)
           X = (40,80), y = (160)
       and shuffled before return."""
    x0 = np.concatenate([
        np.random.normal(20, 2, size=n),
        np.random.normal(20, 2, size=n),
        np.random.normal(40, 2, size=n),
        np.random.normal(40, 2, size=n),
    ])
    x1 = np.concatenate([
        np.random.normal(60, 2, size=n),
        np.random.normal(80, 2, size=n),
        np.random.normal(60, 2, size=n),
        np.random.normal(80, 2, size=n),
    ])
    y = np.concatenate([
        np.random.normal(100, 2, size=n),
        np.random.normal(120, 2, size=n),
        np.random.normal(140, 2, size=n),
        np.random.normal(160, 2, size=n),
    ])
    
    # Shuffle data before returning
    data = np.stack((x0, x1, y), axis=1).astype(np.float32)
    np.random.shuffle(data)
    
    # 3D plot of dataset
    if plot:
        plot_data(data[:,:2], data[:,2:])
    
    # Able to specify 2D-X and 1D-y or vice versa
    return (data[:,:2], data[:,2:]) \
            if xy_features == (2,1) else \
            (data[:,:1], data[:,1:])