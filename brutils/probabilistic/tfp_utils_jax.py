import time
import functools
from functools import partial
from brutils import utility as ut

import numpy as np
import jax
from jax import random

SEED = random.PRNGKey(1234)
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
import numpyro as npr

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

# from tensorflow import nest
JointDistributionPinned.sample = JointDistributionPinned.sample_unpinned

root = tfd.JointDistributionCoroutine.Root
mcmc = tfp.mcmc
DTYPE = "float32"
from packaging.version import Version, parse

TARGET_VERSION = Version("2.5")

PARAMETERS = {}

if not hasattr(tfd.Distribution, "old_sample"):
    tfd.Distribution.old_sample = tfd.Distribution.sample


def get_bijected_samples(model, bijector, num_chains):
    samples = get_samples(model, num_chains)
    if not isinstance(bijector, list):
        return bijector.inverse(samples)
    else:
        return samples
        # return tf.nest.pack_sequence_as(model.event_shape, [bij.inverse(s) for bij, s in zip(bijector, samples)])


def get_samples(model, size, seed=SEED):
    if hasattr(model, "sample"):
        samples = model.sample(size, seed=SEED)
    elif hasattr(model, "sample_unpinned"):
        samples = model.sample_unpinned(size, seed=SEED)
    else:
        raise ValueError("Invalid Model")
    return samples


def get_bijectors_from_samples(samples, unconstraining_bijectors, batch_axes):
    """Fit bijectors to the samples of a distribution.

    This fits a diagonal covariance multivariate Gaussian transformed by the
    `unconstraining_bijectors` to the provided samples. The resultant
    transformation can be used to precondition MCMC and other inference methods.
    """
    state_std = [
        jnp.std(bij.inverse(x), axis=batch_axes)
        for x, bij in zip(samples, unconstraining_bijectors)
    ]
    state_mu = [
        jnp.mean(bij.inverse(x), axis=batch_axes)
        for x, bij in zip(samples, unconstraining_bijectors)
    ]
    return [
        tfb.Chain([cb, tfb.Shift(sh), tfb.Scale(sc)])
        for cb, sh, sc in zip(unconstraining_bijectors, state_mu, state_std)
    ]


def generate_init_state_and_bijectors_from_prior(
    model, nchain, unconstraining_bijectors
):
    """Creates an initial MCMC state, and bijectors from the prior."""
    prior_samples = get_samples(model, 4096)

    bijectors = get_bijectors_from_samples(
        prior_samples, unconstraining_bijectors, batch_axes=0
    )

    # init_state = [
    #     bij(tf.zeros([nchain] + list(s), DTYPE))
    #     for s, bij in zip(model.event_shape, bijectors)
    # ]
    # init_state = tf.nest.pack_sequence_as(model.event_shape, init_state)
    init_state = get_bijected_samples(model, unconstraining_bijectors, nchain)

    return init_state, bijectors


def sample_trace_fn_nuts(_, pkr):
    energy_diff = unnest.get_innermost(pkr, "log_accept_ratio")
    return {
        "target_log_prob": unnest.get_innermost(pkr, "target_log_prob"),
        "n_steps": unnest.get_innermost(pkr, "leapfrogs_taken"),
        "diverging": unnest.get_innermost(pkr, "has_divergence"),
        "energy": unnest.get_innermost(pkr, "energy"),
        "accept_ratio": jnp.minimum(1.0, jnp.exp(energy_diff)),
        "reach_max_depth": unnest.get_innermost(pkr, "reach_max_depth"),
        "acceptance_ratio": unnest.get_innermost(pkr, "is_accepted"),
    }


def sample_trace_fn_hamiltonian(_, pkr):
    energy_diff = unnest.get_innermost(pkr, "log_accept_ratio")
    return {
        "target_log_prob": unnest.get_innermost(pkr, "target_log_prob"),
        "accept_ratio": jnp.minimum(1.0, jnp.exp(energy_diff)),
        "acceptance_ratio": unnest.get_innermost(pkr, "is_accepted"),
    }


def sample_(
    target_model,
    num_chains=4,
    num_results=1000,
    step_size=0.005,
    bijector_fn=None,
    log_likelihood=None,
    num_leapfrog_steps=None,
    num_burnin_steps=None,
    initial_state=None,
    kernel=None,
    trace_fn=None,
    target_accept_prob=0.8,
    initialize_method="simple",
    step_size_fn=None,
    seed=SEED,
):
    num_burnin_steps = (
        num_burnin_steps if num_burnin_steps is not None else num_results // 2
    )
    if step_size_fn is None:
        step_size_fn = lambda hmc: mcmc.SimpleStepSizeAdaptation(
            hmc,
            num_adaptation_steps=int(num_burnin_steps * 0.8),
            target_accept_prob=target_accept_prob,
        )
    if trace_fn is None:
        if num_leapfrog_steps is None:
            trace_fn = sample_trace_fn_nuts
        else:
            trace_fn = sample_trace_fn_hamiltonian
    if bijector_fn is None:
        bijector = [tfb.Identity() for _ in target_model.event_shape]
        # bijector = target_model.experimental_default_event_space_bijector()
    else:
        bijector = bijector_fn()
    if initial_state is not None:
        pass
    elif initialize_method == "simple":
        initial_state = get_bijected_samples(target_model, bijector, num_chains)
    elif initialize_method == "isotropic_normal":
        initial_state, bijector = generate_init_state_and_bijectors_from_prior(
            target_model, num_chains, bijector
        )
    else:
        raise ValueError("initialization method invalid.")
    if kernel is None:
        log_likelihood = (
            target_model.log_prob if log_likelihood is None else log_likelihood
        )
        if num_leapfrog_steps is None:
            sampler = mcmc.NoUTurnSampler(
                log_likelihood,
                step_size=step_size,
            )
        else:
            sampler = mcmc.HamiltonianMonteCarlo(
                log_likelihood,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps,
            )
        kernel = mcmc.TransformedTransitionKernel(sampler, bijector=bijector)
        kernel = step_size_fn(kernel)
    mcmc_samples, sampler_stats = mcmc.sample_chain(
        num_results,
        current_state=initial_state,
        kernel=kernel,
        trace_fn=trace_fn,
        num_burnin_steps=num_burnin_steps,
        seed=SEED,
    )
    return mcmc_samples, sampler_stats


@functools.wraps(sample_)
def sample(*args, **kwargs):
    bijectors = kwargs.pop("bijectors", None)
    kwargs["bijector_fn"] = lambda: bijectors
    mcmc_samples, sampler_stats = sample_(*args, **kwargs)
    print("R-hat:", check_rhat(mcmc_samples))
    return to_az_inference(mcmc_samples, sampler_stats)


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


def check_rhat(trace, *, thresh=None, ignore_nan=False):
    # the following was added when trying to assess lower triangular matrices in LKJ
    # def remove_nan(x):
    #     return tf.where(tf.math.is_nan(x), 0, x)
    # out = tf.nest.map_structure(lambda x: tf.reduce_max(remove_nan(tf.abs(mcmc.potential_scale_reduction(x).v - 1))).numpy(), trace)
    def fn(x):
        y = jnp.abs(mcmc.potential_scale_reduction(x) - 1)
        out = np.array(y).squeeze()
        if ignore_nan and np.isnan(out).sum() > 0:
            logger.warning("some variables have nan values")
            out = np.where(np.isnan(out), 0, out)
        return out.max()

    out = jax.tree.map(fn, trace)
    if thresh is not None:
        out = jax.tree.map(lambda x: x < thresh, out)
    return out


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


@ut.RegisterWithClass(tfd.Distribution)
def sample(self, sample_shape=(), seed=None, name="sample", **kwargs):
    if seed is None:
        seed = jax.random.PRNGKey(int(time.time() * 1e9))
    elif isinstance(seed, int):
        seed = jax.random.PRNGKey(seed)
    return self.old_sample(sample_shape, seed=seed, name=name, **kwargs)


class JointDistributionCoroutine:
    def __init__(self, fun):
        self.fun = fun

    def __call__(self, *args, **kwargs):
        @tfd.JointDistributionCoroutine
        @functools.wraps(self.fun)
        def model():
            yield from self.fun(*args)

        if len(kwargs):
            return model.experimental_pin(**kwargs)
        return model


class NumpyroDist(tfd.Distribution):
    def __init__(self, dist, name="numpyro_dist"):
        self.dist_ = dist
        cls = type(tfd.Normal(0, 1).reparameterization_type)
        self._reparameterization_type = (
            cls("FULLY_REPARAMETERIZED")
            if len(dist.reparametrized_params)
            else cls("NOT_REPARAMETERIZED")
        )
        self._name = "numpyro_dist" if name is None else name
        self._dtype = np.float32

    def __repr__(self):
        return repr(self.dist_)

    def _batch_shape(self):
        return self.dist_.batch_shape

    def _event_shape(self):
        return self.dist_.event_shape

    def sample(
        self,
        sample_shape=(),
        seed=None,
        name="sample",
        **kwargs,
    ):
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        return self.dist_.sample(seed, tuple(sample_shape))

    def log_prob(self, x):
        return self.dist_.log_prob(x)


@ut.RegisterWithClass(npr.distributions.Distribution)
def to_tfp(self, name=None):
    return NumpyroDist(self, name=name)


def rt(self):
    return root(self)


tfd.Distribution.root = property(rt)
