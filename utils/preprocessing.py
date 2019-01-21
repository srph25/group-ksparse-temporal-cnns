import numpy as np
from sklearn.utils.extmath import svd_flip, _incremental_mean_and_var
import threading


class ZCAWhitening():
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        
    def fit_transform(self, X):
        if X.ndim == 4:
            axis = 0
        elif X.ndim == 5:
            axis = (0, 1)
        self.mean = np.mean(X, axis=axis, keepdims=True)
        X = (X - self.mean)
        X_flat = np.reshape(X, (-1, np.prod(X.shape[-3:])))
        sigma = np.dot(X_flat.T, X_flat) / X_flat.shape[0]
        u, s, _ = np.linalg.svd(sigma)
        s_inv = np.diag(1. / np.sqrt(s + self.epsilon))
        self.principal_components = (u.dot(s_inv)).dot(u.T)
        X_white = np.reshape(np.dot(X_flat, self.principal_components), X.shape)
        return X_white

    def transform(self, X):        
        X = (X - self.mean)
        X_flat = np.reshape(X, (-1, np.prod(X.shape[-3:])))
        X_white = np.reshape(np.dot(X_flat, self.principal_components), X.shape)
        return X_white


class IncrementalZCAWhitening():
    def __init__(self, epsilon=0.1, max_components=3000, pad_value=-1, n_samples_train=50000):
        self.epsilon = epsilon
        self.max_components = max_components
        self.pad_value = pad_value
        self.n_samples_train = n_samples_train
        self.n_samples_seen = 0
        self.mean = 0.
        self.var = 0.
        self.lock = threading.Lock()
        
    def partial_fit_transform(self, X):
        shp = X.shape
        if X.ndim == 4:
            X = np.reshape(X, (shp[0], -1))
        elif X.ndim == 5:
            X = np.reshape(X, (shp[0] * shp[1], -1))
        whr = np.where(np.any(X != self.pad_value, axis=1))[0]
        if len(whr) > 0:
            if self.n_samples_seen < self.n_samples_train:
                self.lock.acquire()
                try:
                    # Update stats - they are 0 if this is the fisrt step
                    col_mean, col_var, n_total_samples = \
                        _incremental_mean_and_var(
                            X[whr], last_mean=self.mean, last_variance=self.var, last_sample_count=np.repeat(self.n_samples_seen, X[whr].shape[1]))
                    n_total_samples = n_total_samples[0]
                    if self.n_samples_seen == 0:
                        X[whr] = X[whr] - col_mean
                        _X = X[whr]
                    else:
                        col_batch_mean = np.mean(X[whr], axis=0)
                        X[whr] = X[whr] - col_batch_mean
                        # Build matrix of combined previous basis and new data
                        mean_correction = np.sqrt((self.n_samples_seen * X[whr].shape[0]) / n_total_samples) * (self.mean - col_batch_mean)
                        _X = np.vstack((self.singular_values.reshape((-1, 1)) * self.components, X[whr], mean_correction))
                    U, S, V = np.linalg.svd(_X, full_matrices=False)
                    U, V = svd_flip(U, V, u_based_decision=False)
                    explained_variance = S ** 2 / (n_total_samples - 1)
                    self.n_samples_seen = n_total_samples
                    self.components = V[:self.max_components]
                    self.singular_values = S[:self.max_components]
                    self.mean = col_mean
                    self.var = col_var
                    self.explained_variance = explained_variance[:self.max_components]
                finally:
                    self.lock.release()
            else:
                X[whr] = X[whr] - self.mean
            X[whr] = np.dot((np.dot(X[whr], self.components.T) / np.sqrt(self.explained_variance + self.epsilon)), self.components)
        return np.reshape(X, shp)
        
    def transform(self, X):
        shp = X.shape
        if X.ndim == 4:
            X = np.reshape(X, (shp[0], -1))
        elif X.ndim == 5:
            X = np.reshape(X, (shp[0] * shp[1], -1))
        whr = np.where(np.any(X != self.pad_value, axis=1))[0]
        if len(whr) > 0:
            X[whr] = X[whr] - self.mean
            X[whr] = np.dot((np.dot(X[whr], self.components.T) / np.sqrt(self.explained_variance + self.epsilon)), self.components)
        return np.reshape(X, shp)

