import functools
from functools import partial
from brutils import utility as ut

import numpy as np
import jax
from jax import random

SEED = random.PRNGKey(1234)
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels

import xarray as xr
import arviz as az
import logging

logger = logging.getLogger("tfp_utils")
from tensorflow_probability.substrates.jax.internal.structural_tuple import structtuple
from tensorflow_probability.substrates.jax.internal import samplers
from tensorflow_probability.substrates.jax.experimental.distributions.joint_distribution_pinned import (
    JointDistributionPinned,
)
from brutils.utility import RegisterWithClass
from tensorflow_probability.substrates.jax.internal import unnest

JointDistributionPinned.sample = JointDistributionPinned.sample_unpinned

root = tfd.JointDistributionCoroutine.Root
mcmc = tfp.mcmc
DTYPE = "float32"
from packaging.version import Version, parse

TARGET_VERSION = Version("2.5")

PARAMETERS = {}


def add_loglikelihood_to_inference_data(pinnedModel, posteriorSamples):
    dists = pinnedModel.distribution.sample_distributions(**posteriorSamples.dict)[0]

    def get_pinned_dist(k):
        d = getattr(dists, k)
        if hasattr(d, "distribution"):
            d = d.distribution
        return d

    sl = {k: get_pinned_dist(k).log_prob(v) for k, v in pinnedModel.pins.items()}
    sp = posteriorSamples.copy()
    sp.add_groups({"log_likelihood": sl})
    return sp


def to_darray(v, extra_dims=None):
    extra_dims = extra_dims if extra_dims is not None else {}
    return xr.DataArray(
        np.array(v), dims=["draw", "chain"] + list(extra_dims), coords=extra_dims
    )


def to_az_inference(mcmc_samples, sampler_stats=None):
    if sampler_stats is None:
        sample_stats = {}
    else:
        sample_stats = {
            k: np.swapaxes(sampler_stats[k], 1, 0)
            # for k in ["target_log_prob", "diverging", "accept_ratio", "n_steps", "acceptance_ratio"]
            for k in sampler_stats
        }
    return az.from_dict(
        posterior={
            k: np.swapaxes(np.array(v), 1, 0) for k, v in mcmc_samples._asdict().items()
        },
        sample_stats=sample_stats,
    )


def to_az_inference_with_coords(trace, coords):
    dic = {}
    for k, v in trace._asdict().items():
        dic[k] = to_darray(v, coords.get(k, None))
    # res = xr.Dataset(dic).rename_dims()
    # return az.InferenceData(posterior=res)


def to_az_inference_old(trace):
    return az.from_dict(
        posterior={
            k: np.swapaxes(np.array(v), 1, 0) for k, v in trace._asdict().items()
        },
    )


class SmoothLinear:
    def __init__(self, n_tp, n_changepoints):
        self.t = np.linspace(0, 1, n_tp).astype("float32")
        self.s = np.linspace(0, 1, n_changepoints + 2)[1:-1].astype("float32")
        self.A = (self.t[:, None] > self.s).astype("float32")

    def __getitem__(self, s):
        out = SmoothLinear(10, 10)
        out.t = self.t[s]
        out.A = self.A[s]
        out.s = self.s
        return out

    def __call__(self, k, m, slopes):
        """
        len(slopes) == n_changepoints
        """
        growth = (k + jnp.einsum("ij,...j->...i", self.A, slopes)) * self.t
        offset = m + jnp.einsum(
            "ij,...j->...i", self.A, jnp.einsum("j,...j->...j", -self.s, slopes)
        )
        return growth + offset


def gen_fourier_basis(t, max_period, num_basis):
    x = 2 * np.pi * (np.arange(num_basis) + 1) * t[:, None] / max_period
    return np.concatenate((np.cos(x), np.sin(x)), axis=1)


def asdict(d):
    return {k: v.values for k, v in d.posterior.data_vars.items()}


@ut.RegisterWithClass(az.data.inference_data.InferenceData)
def mean_asdict(d, axis=(0, 1)):
    return {k: v.values.mean(axis) for k, v in d.posterior.data_vars.items()}


@ut.RegisterWithClass(tfd.Distribution)
def expand(self, size, name=None):
    name = name or self.name
    return tfd.Sample(self, size, name=name)


@ut.RegisterWithClass(tfd.Distribution)
def Sample(self, size, name=None):
    name = name or self.name
    return tfd.Sample(self, size, name=name)


def Ind(d, reinterpreted_batch_ndims=1, **kwargs):
    return tfd.Independent(
        d, reinterpreted_batch_ndims=reinterpreted_batch_ndims, **kwargs
    )


@ut.RegisterWithClass(tfd.Distribution)
def event(self, n_dims=1, name=None):
    name = name or self.name
    return Ind(self, n_dims, name=name)


def rt(self):
    return root(self)


tfd.Distribution.root = property(rt)
