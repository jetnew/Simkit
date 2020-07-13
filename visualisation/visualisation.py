import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from scipy.stats import binned_statistic


def plot_surface(X1, X2, bins=20):
    """Plot the density function based on X1 and X2."""
    pdf, x1, x2 = np.histogram2d(X1, X2, bins=bins, density=True)
    xmesh, ymesh = np.meshgrid(x1[:-1], x2[:-1])
    ax = plt.axes(projection='3d')
    ax.plot_surface(xmesh, ymesh, pdf, cmap='viridis')
    plt.show()


def plot_prob_fixed(model, X_fixed, y_range, count=20):
    """Plot probability density function given X_fixed across y_range."""
    assert hasattr(model, 'prob')
    X = np.stack([np.full(count, fill_value=x) for x in X_fixed], axis=1)
    y = np.linspace(y_range[0], y_range[1], count)

    y_prob = model.prob(X, y[:, np.newaxis])
    plt.title(f"X={X_fixed}, y={y_range}")
    plt.plot(y, y_prob)
    plt.show()
    

def plot_violin(density1, density2, feature="", ax=None):
    """Plot a violin to compare 2 densities.
    Given 2 sets of data, plot a violin plot to compare the probability density function. 
    """
    df = pd.DataFrame({
        'density': np.concatenate([density1, density2]),
        'feature': [feature] * len(density1) * 2,
        'distribution': ["actual"] * len(density1) + ["fitted"] * len(density2)
    })
    if ax is None:
        sns.violinplot(x="feature", y="density", hue="distribution", data=df, split=True)
        plt.show()
    else:
        sns.violinplot(x="feature", y="density", hue="distribution", data=df, split=True, ax=ax)


def plot_prob_violin(model, X, y, X_fixed, X_tol):
    """Given X, plot a violin of y based on X within a tolerance, to compare between fitted and actual.
    X_tol represents the tolerance to which X data is binned to get the corresponding y.
    X_fixed represents the X fed into the model to sample y.
    """
    assert hasattr(model, 'predict')
    # Get indices where X lies within X_tol of X_fixed
    X_idx = [np.where(np.abs(X[:,i] - X_fixed[i]) < X_tol[i])[0] for i in range(X.shape[1])]
    idx = reduce(np.intersect1d, X_idx)
    assert len(idx) > 0

    # Sample based on X_fixed
    X_fix = np.stack([np.full(len(idx), fill_value=x) for x in X_fixed], axis=1)
    y_hat = model.predict(X[idx])

    # Plot violins depending on y dimension
    for y_dim in range(y.shape[1]):
        plot_violin(y[idx, y_dim], y_hat[:,y_dim], feature=f"X={X_fixed}, Count={len(idx)}")
        
        
def plot_binned_violin(model, X, y, bins=5):
    """Given X and y, plot a violin of y based on binned X, to compare between fitted and actual.
    For the actual distribution, bin X and plot violin plots for the density of y.
    For the fitted distribution, input the mean of binned X and plot violin plots for the density of sampled y.
    """
    assert y.shape[1] == 1
    assert hasattr(model, 'predict')
    # X is 2D
    if X.shape[1] == 2:
        # Bin X values to get probabilities of y given X
        X0_bin = binned_statistic(X[:,0], X[:,0], bins=bins)
        X1_bin = binned_statistic(X[:,1], X[:,1], bins=bins)
        fig, ax = plt.subplots(bins, bins, figsize=(20,12))

        for x0 in range(bins):
            for x1 in range(bins):
                # Get indices where X belong to each bin
                idx = np.intersect1d(np.where(X0_bin[2]==x0+1)[0], np.where(X1_bin[2]==x1+1)[0])
                if len(idx) > 0:
                    # Given the mean
                    y_hat = model.predict(X[idx])
                    plot_violin(y[idx,0], y_hat[:,0], f"X=({round(X0_bin[0][x0],2)},{round(X1_bin[0][x1],2)}), Count={len(idx)}", ax[x0,x1])
    # X is 1D
    if X.shape[1] == 1:
        X0_bin = binned_statistic(X[:,0], X[:,0], bins=bins)
        fig, ax = plt.subplots(1, bins, figsize=(20,12))
        for x0 in range(bins):
            idx = np.where(X0_bin[2]==x0+1)[0]
            if len(idx) > 0:
                y_hat = model.predict(X[idx])
                plot_violin(y[idx,0], y_hat[:,0], f"X=({round(X0_bin[0][x0],2)},), Count={len(idx)}", ax[x0])