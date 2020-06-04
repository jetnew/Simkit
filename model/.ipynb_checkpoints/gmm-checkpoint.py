import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


class GMM:
    def fit(self, X, max_components=30):
        """Fits and returns GMM with lowest BIC.
        X - np.array
        max_components - int, upper limit of components to fit GMM with.
        
        E.g.
        X = np.array([[1,1], [2,2], [3,3]])
        gmm = fit_GMM(X, max_components=3)
        """
        n_components = np.arange(1, max_components)
        models = [GaussianMixture(n, covariance_type='full').fit(X)
                  for n in n_components]
        bics = [m.bic(X) for m in models]
        self.bic = min(bics)
        self.gmm = models[bics.index(self.bic)]
        return self.gmm

    def plot(self, X):
        """Plots the clusters of GMM given data.
        If 1D, plots a histogram.
        If 2D, plots a scatterplot.
        If >2D, performs PCA with 2 components then plots scatterplot.
        """
        y_hat = self.gmm.predict(X)
        if X.shape[1] == 1:
            plt.hist(X)
            plt.show()
        else:
            X_hat = PCA(n_components=2).fit_transform(X) if X.shape[1] > 2 else X
            plt.scatter(X_hat[:,0], X_hat[:,1], c=y_hat, cmap='viridis')
            plt.show()

    def sample(self, sample_size,  c=None):
        """Given a GMM and component number, sample from the component.
        sample_size - int
        gmm - GaussianMixture
        c - int, component no.
        """
        m = self.gmm
        if c is not None:
            m = GaussianMixture(n_components=1)
            m.weights_ = [1]
            m.covariances_ = np.expand_dims(self.gmm.covariances_[c], axis=0)
            m.means_ = np.expand_dims(self.gmm.means_[c], axis=0)
            m.precisions_ = np.expand_dims(self.gmm.precisions_[c], axis=0)
            m.precisions_cholesky_ = np.expand_dims(self.gmm.precisions_cholesky_[c], axis=0)
        return m.sample(sample_size)

if __name__ == "__main__":
    """Scikit-learn Implementation"""
    from model.gmm import GMM

    # Fit a GMM and assign data to components
    gmm = GMM()
    gmm.fit(X, max_components=30)
    gmm.plot(X)

    # Sample 100 from component 0
    xhat, yhat = gmm.sample(100, c=0)
    plt.scatter(xhat[:,0], xhat[:,1])
    plt.show()
