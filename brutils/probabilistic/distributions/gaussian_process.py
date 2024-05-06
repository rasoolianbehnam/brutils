from functools import reduce

import numpy as np
from dataclasses import dataclass, field
from tensorflow_probability import distributions as tfd
from brutils.tf_registers import *


def reshape_inputs(x1, x2):
    return x1[:, None, :], x2[None, :, :]


@dataclass
class GaussianProcess_:
    observation_std: float = field()

    def mean(self, x1):
        return np.zeros((x1.shape[1], 1))

    def Kernel(self, x1, x2):
        x1, x2 = self.transform(x1), self.transform(x2)
        n, d = x1.shape
        assert x1.shape[1] == d
        m = x2.shape[0]
        return self.kernel(x1, x2).reshape([n, m])

    def phi(self, x):
        return x
    
    def transform(self, x):
        return x

    def kernel(self, x1, x2):
        return self.phi(x1) @ self.phi(x2).T

    def fit(self, X, y):
        n = len(X)
        self.X = X
        self.y = y
        self.right = self.Kernel(X, X)
        self.F = np.linalg.inv(self.kXX + np.eye(n) * self.observation_std ** 2)
        return self

    def fit_reg(self, x, y, prior_sigma):
        phi = self.phi(x)
        d = phi.shape[1]
        sigma = prior_sigma**-2 * np.eye(d)
        cov = np.linalg.inv(sigma + self.observation_std**-2 * phi.T @ phi)
        sig = np.linalg.cholesky(cov).reshape(d, d)
        mu = (phi.T @ y).reshape(-1)
        return tfd.MultivariateNormalTriL(mu, sig)

    def predict(self, x):
        X, y = self.X, self.y
        kXx = self.Kernel(X, x)
        kxx = self.Kernel(x, x)
        kxX = kXx.T
        m = self.mean(x)
        m = m + kxX @ self.F @ (y - m)
        s = kxx - kxX @ self.F @ kXx
        return tfd.MultivariateNormalDiag(m.flatten(), np.sqrt(np.diag(s)))

    def fit_predict(self, x, y, xx):
        return self.fit(x, y).predict(xx)


@dataclass
class GaussianProcess(GaussianProcess_):
    observation_std: float = field()

    def mean(self, x1):
        return 0 # tf.zeros([x1.shape[1], 1], dtype='float64')

    def fit(self, X, y):
        n = len(X)
        self.X = X
        self.y = y
        self.kXX = self.Kernel(X, X)
        self.F = tf.linalg.inv(self.kXX + tf.eye(n, dtype='float64') * self.observation_std ** 2)
        return self

    def predict(self, x):
        X, y = self.X, self.y
        kXx = self.Kernel(X, x)
        kxx = self.Kernel(x, x)
        kxX = kXx.T
        m = self.mean(x)
        m = m + kxX @ self.F @ (y - m)
        s = kxx - kxX @ self.F @ kXx
        return tfd.MultivariateNormalDiag(m.reshape(-1), tf.sqrt(tf.linalg.diag_part(s)))


def mixed_gp(*gps, operator):
    def Kernel(x1, x2):
        return reduce(operator, [a.Kernel(x1, x2) for a in gps])
    return Kernel
