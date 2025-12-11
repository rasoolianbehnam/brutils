import jax
from jax import vmap
from dataclasses import dataclass
from dataclasses import dataclass
from typing import Callable
from itertools import product
from numpy.polynomial.hermite_e import hermegauss

jnp = jax.numpy


@dataclass
class EKF:
    gaussian_expectation: Callable = lambda f, m, P: f(m)
    gaussian_cross_covariance: Callable = (
        lambda f, g, m, P: jax.jacfwd(f)(m) @ P @ jax.jacfwd(g)(m).T
    )
    transform: Callable = None

    def __post_init__(self):
        self.transform = self._transform

    def _transform(self, f, m, P):
        m_post = self.gaussian_expectation(f, m, P)
        P_post = self.gaussian_cross_covariance(f, f, m, P)
        C_post = self.gaussian_cross_covariance(lambda x: x, f, m, P)
        return m_post, P_post, C_post


@dataclass
class SigmaPointMethods:
    # Compute expectation value of f over N(m, P)
    def _gaussian_expectation(self, f, m, P):
        w_mean, _, sigmas = self.compute_weights_and_sigmas(m, P)
        return jnp.tensordot(w_mean, vmap(f)(sigmas), axes=1)

    # Compute cross covariance of f and g over N(m, P)
    def _gaussian_cross_covariance(self, f, g, m, P):
        _, w_cov, sigmas = self.compute_weights_and_sigmas(m, P)
        _outer = vmap(lambda x, y: jnp.atleast_2d(x).T @ jnp.atleast_2d(y), 0, 0)
        f_mean, g_mean = self.gaussian_expectation(f, m, P), self.gaussian_expectation(
            g, m, P
        )
        return jnp.tensordot(
            w_cov, _outer(vmap(f)(sigmas) - f_mean, vmap(g)(sigmas) - g_mean), axes=1
        )

    # Return sigma points pre- and post-transform
    def _sigma_points(self, f, m, P):
        *_, sigmas = self.compute_weights_and_sigmas(m, P)
        sigmas_prop = vmap(f)(sigmas)
        return sigmas, sigmas_prop

    # Compute (m,P,C) over transformed sigma points
    def _transform(self, f, m, P):
        m_post = self._gaussian_expectation(f, m, P)
        P_post = self._gaussian_cross_covariance(f, f, m, P)
        C_post = self._gaussian_cross_covariance(lambda x: x, f, m, P)
        return m_post, P_post, C_post


@dataclass
class UKF(SigmaPointMethods):
    alpha: float = jnp.sqrt(3)
    beta: float = 2
    kappa: float = 1
    compute_weights_and_sigmas: Callable = lambda x, y: (0, 0, 0)
    gaussian_expectation: Callable = None
    gaussian_cross_covariance: Callable = None
    sigma_points: Callable = None
    transform: Callable = None

    def __post_init__(self):
        self.compute_weights_and_sigmas = self._compute_weights_and_sigmas
        self.gaussian_expectation = super()._gaussian_expectation
        self.gaussian_cross_covariance = super()._gaussian_cross_covariance
        self.sigma_points = super()._sigma_points
        self.transform = super()._transform

    def _compute_weights_and_sigmas(self, m, P):
        n = len(m)
        lamb = self.alpha**2 * (n + self.kappa) - n
        # Compute weights
        factor = 1 / (2 * (n + lamb))
        w_mean = jnp.concatenate(
            (jnp.array([lamb / (n + lamb)]), jnp.ones(2 * n) * factor)
        )
        w_cov = jnp.concatenate(
            (
                jnp.array([lamb / (n + lamb) + (1 - self.alpha**2 + self.beta)]),
                jnp.ones(2 * n) * factor,
            )
        )
        # Compute sigmas
        distances = jnp.sqrt(n + lamb) * jnp.linalg.cholesky(P)
        sigma_plus = jnp.array([m + distances[:, i] for i in range(n)])
        sigma_minus = jnp.array([m - distances[:, i] for i in range(n)])
        sigmas = jnp.concatenate((jnp.array([m]), sigma_plus, sigma_minus))
        return w_mean, w_cov, sigmas


@dataclass
class GHKF(SigmaPointMethods):
    """
    Lightweight container for Gauss-Hermite Kalman filter/smoother parameters.
    """

    order: int = 10
    compute_weights_and_sigmas: Callable = lambda x, y: (0, 0, 0)
    gaussian_expectation: Callable = None
    gaussian_cross_covariance: Callable = None
    sigma_points: Callable = None
    transform: Callable = None

    def __post_init__(self):
        self.compute_weights_and_sigmas = self._compute_weights_and_sigmas
        self.gaussian_expectation = super()._gaussian_expectation
        self.gaussian_cross_covariance = super()._gaussian_cross_covariance
        self.sigma_points = super()._sigma_points
        self.transform = super()._transform

    def _compute_weights_and_sigmas(self, m, P):
        n = len(m)
        samples_1d, weights_1d = hermegauss(self.order)
        weights_1d /= weights_1d.sum()
        weights = jnp.prod(jnp.array(list(product(weights_1d, repeat=n))), axis=1)
        unit_sigmas = jnp.array(list(product(samples_1d, repeat=n)))
        sigmas = m + vmap(jnp.matmul, [None, 0], 0)(jnp.linalg.cholesky(P), unit_sigmas)
        return weights, weights, sigmas
