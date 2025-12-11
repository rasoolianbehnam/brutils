import functools
from functools import partial
from brutils import utility as ut

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr
import arviz as az
import logging

logger = logging.getLogger("tfp_utils")
from tensorflow_probability.python.internal.structural_tuple import structtuple
from tensorflow_probability.python.experimental.distributions.joint_distribution_pinned import (
    JointDistributionPinned,
)
from brutils.utility import RegisterWithClass
from tensorflow_probability.python.internal import unnest

JointDistributionPinned.sample = JointDistributionPinned.sample_unpinned

keras = tf.keras
tfd = tfp.distributions
tfb = tfp.bijectors
tfl = tfp.layers
tfli = tf.linalg
Root = tfd.JointDistributionCoroutine.Root
mcmc = tfp.mcmc
DTYPE = "float32"
from packaging.version import Version, parse

TARGET_VERSION = Version("2.5")
PI = tf.constant(np.pi, dtype="float32")

if Version(tf.__version__) < TARGET_VERSION:
    tf_function = partial(tf.function, autograph=False)
else:
    tf_function = partial(tf.function, jit_compile=True, autograph=False)

PARAMETERS = {}


def clear_param_store():
    global PARAMETERS
    PARAMETERS = {}


def Variable(name, value, bijector=None, overwrite=False):
    if name in PARAMETERS and not overwrite:
        raise ValueError(f"Name {name} already in param store!")
    if bijector is None:
        v = tf.Variable(value)
        PARAMETERS[name] = v
    else:
        v = tfp.util.TransformedVariable(value, bijector)
        PARAMETERS[name] = v.trainable_variables
    return v


def trainable_variables():
    return tuple(PARAMETERS.values())


def get_bijected_samples(model, bijector, num_chains):
    samples = get_samples(model, num_chains)
    if not isinstance(bijector, list):
        return bijector.inverse(samples)
    else:
        return samples
        # return tf.nest.pack_sequence_as(model.event_shape, [bij.inverse(s) for bij, s in zip(bijector, samples)])


def get_samples(model, size):
    if hasattr(model, "sample"):
        samples = model.sample(size)
    elif hasattr(model, "sample_unpinned"):
        samples = model.sample_unpinned(size)
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
        tf.math.reduce_std(bij.inverse(x), axis=batch_axes)
        for x, bij in zip(samples, unconstraining_bijectors)
    ]
    state_mu = [
        tf.math.reduce_mean(bij.inverse(x), axis=batch_axes)
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


@tf_function
def sample_trace_fn_nuts(_, pkr):
    energy_diff = unnest.get_innermost(pkr, "log_accept_ratio")
    return {
        "target_log_prob": unnest.get_innermost(pkr, "target_log_prob"),
        "n_steps": unnest.get_innermost(pkr, "leapfrogs_taken"),
        "diverging": unnest.get_innermost(pkr, "has_divergence"),
        "energy": unnest.get_innermost(pkr, "energy"),
        "accept_ratio": tf.minimum(1.0, tf.exp(energy_diff)),
        "reach_max_depth": unnest.get_innermost(pkr, "reach_max_depth"),
        "acceptance_ratio": unnest.get_innermost(pkr, "is_accepted"),
    }


@tf_function
def sample_trace_fn_hamiltonian(_, pkr):
    energy_diff = unnest.get_innermost(pkr, "log_accept_ratio")
    return {
        "target_log_prob": unnest.get_innermost(pkr, "target_log_prob"),
        "accept_ratio": tf.minimum(1.0, tf.exp(energy_diff)),
        "acceptance_ratio": unnest.get_innermost(pkr, "is_accepted"),
    }


@tf_function
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
    )
    return mcmc_samples, sampler_stats


@functools.wraps(sample_)
def sample(*args, **kwargs):
    bijectors = kwargs.pop("bijectors", None)
    kwargs["bijector_fn"] = lambda: bijectors
    mcmc_samples, sampler_stats = sample_(*args, **kwargs)
    print("R-hat:", check_rhat(mcmc_samples))
    return to_az_inference(mcmc_samples, sampler_stats)


def add_loglikelihood_to_inference_data(pinnedModel, posteriorSamples, inplace=False):
    dists = pinnedModel.distribution.sample_distributions(**posteriorSamples.dict)[0]

    def get_pinned_dist(k):
        d = getattr(dists, k)
        if hasattr(d, "distribution"):
            d = d.distribution
        return d

    sl = {k: get_pinned_dist(k).log_prob(v) for k, v in pinnedModel.pins.items()}
    if not inplace:
        sp = posteriorSamples.copy()
    else:
        sp = posteriorSamples
    sp.add_groups({"log_likelihood": sl})
    return sp


def to_darray(v, extra_dims=None):
    extra_dims = extra_dims if extra_dims is not None else {}
    return xr.DataArray(
        v.numpy(), dims=["draw", "chain"] + list(extra_dims), coords=extra_dims
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
            k: np.swapaxes(v.numpy(), 1, 0) for k, v in mcmc_samples._asdict().items()
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
        posterior={k: np.swapaxes(v.numpy(), 1, 0) for k, v in trace._asdict().items()},
    )


def check_rhat(trace, *, thresh=None, ignore_nan=False):
    # the following was added when trying to assess lower triangular matrices in LKJ
    # def remove_nan(x):
    #     return tf.where(tf.math.is_nan(x), 0, x)
    # out = tf.nest.map_structure(lambda x: tf.reduce_max(remove_nan(tf.abs(mcmc.potential_scale_reduction(x).v - 1))).numpy(), trace)
    def fn(x):
        y = tf.abs(mcmc.potential_scale_reduction(x).v - 1)
        out = y.numpy().squeeze()
        if ignore_nan and np.isnan(out).sum() > 0:
            logger.warning("some variables have nan values")
            out = np.where(np.isnan(out), 0, out)
        return out.max()

    out = tf.nest.map_structure(fn, trace)
    if thresh is not None:
        out = tf.nest.map_structure(lambda x: x < thresh, out)
    return out


class SmoothLinear:
    def __init__(self, n_tp, n_changepoints):
        self.t = np.linspace(0, 1, n_tp).astype("float32")
        self.s = np.linspace(0, 1, n_changepoints + 2)[1:-1].astype("float32")
        self.A = tf.cast(self.t[:, None] > self.s, tf.float32)

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
        growth = (k + tf.einsum("ij,...j->...i", self.A, slopes)) * self.t
        offset = m + tf.einsum(
            "ij,...j->...i", self.A, tf.einsum("j,...j->...j", -self.s, slopes)
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


def random_field_surrogate_posterior(
    target,
    event_shape=None,
    bijectors="experimental_default_event_space_bijector",
    init_scale=1,
    operators="diag",
):
    """
    :param target:
    :param init_scale:
    :param operators: Example operators for a model with 2 block variables representing full covariance
        operators = (
            (tf.linalg.LinearOperatorLowerTriangular,),
            (tf.linalg.LinearOperatorFullMatrix, tf.linalg.LinearOperatorLowerTriangular),
        )
    :return:
    """
    event_shape = target.event_shape if event_shape is None else event_shape
    event_sizes = get_event_size(event_shape)
    baseDistribution = tfd.JointDistributionSequential(
        [tfd.Sample(tfd.Normal(0, init_scale), x) for x in event_sizes]
    )
    scale_bijector = get_block_scale_bijector(event_shape, operators)

    shift_bijector = get_loc_bijector(event_shape)
    trainable_variables = (
        scale_bijector.trainable_variables + shift_bijector.trainable_variables
    )

    event_space_bijector = []
    if bijectors == "experimental_default_event_space_bijector" and target is not None:
        event_space_bijector = [
            target.experimental_default_event_space_bijector(),
            tfb.Restructure(
                tf.nest.pack_sequence_as(event_shape, range(len(event_shape)))
            ),
        ]
    elif bijectors is not None:
        event_space_bijector = [
            tfb.Restructure(
                tf.nest.pack_sequence_as(event_shape, range(len(event_shape)))
            ),
            tfb.JointMap(bijectors),
        ]

    bijector = tfb.Chain(
        [
            *event_space_bijector,
            tfb.JointMap(
                [
                    tfb.Reshape(s)
                    for s in [tf.constant(x, dtype="int32") for x in event_shape]
                ]
            ),
            shift_bijector,
            scale_bijector,
        ]
    )
    surrogatePosterior = bijector(baseDistribution)
    return surrogatePosterior, trainable_variables


def get_event_size(event_shape):
    event_sizes = [int(np.prod(x)) for x in event_shape]
    return event_sizes


def get_loc_bijector(event_shape):
    event_sizes = [int(np.prod(x)) for x in event_shape]
    shift_bijector = tfb.JointMap(
        [tfb.Shift(tf.Variable(tf.random.normal([s]))) for s in event_sizes]
    )
    return shift_bijector


def get_block_scale_bijector(event_shape, operators):
    event_sizes = [int(np.prod(x)) for x in event_shape]
    if operators == "tril":
        operators = tuple(
            [
                (tf.linalg.LinearOperatorFullMatrix,) * i
                + (tf.linalg.LinearOperatorLowerTriangular,)
                for i in range(len(event_sizes))
            ]
        )
        scale_bijector = tfb.ScaleMatvecLinearOperatorBlock(
            tfp.experimental.vi.util.build_trainable_linear_operator_block(
                operators, event_sizes
            )
        )
    elif operators == "diag":
        scale_bijector = tfb.JointMap(
            [
                tfb.ScaleMatvecLinearOperator(
                    tf.linalg.LinearOperatorDiag(tf.Variable(tf.random.normal([s])))
                )
                for s in event_sizes
            ]
        )
    else:
        scale_bijector = tfb.ScaleMatvecLinearOperatorBlock(
            tfp.experimental.vi.util.build_trainable_linear_operator_block(
                operators, event_sizes
            )
        )
    return scale_bijector


def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_kernel = keras.Sequential(
        [
            tfl.DistributionLambda(
                lambda t: tfd.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n) * 5
                )
            )
        ]
    )
    return prior_kernel


def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_kernel = keras.Sequential(
        [
            tfl.VariableLayer(tfl.MultivariateNormalTriL.params_size(n), dtype=dtype),
            tfl.MultivariateNormalTriL(n),
        ]
    )
    return posterior_kernel


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


# a NUTS sampling routine with simple tuning
from tensorflow_probability.python.internal import unnest
from tensorflow_probability.python.internal import samplers


def run_mcmc_simple(
    n_draws,
    joint_dist,
    n_chains=4,
    num_adaptation_steps=1000,
    return_compiled_function=False,
    target_log_prob_fn=None,
    bijector=None,
    init_state=None,
    seed=None,
    **pins,
):
    joint_dist_pinned = joint_dist.experimental_pin(**pins) if pins else joint_dist
    if bijector is None:
        bijector = joint_dist_pinned.experimental_default_event_space_bijector()
    if target_log_prob_fn is None:
        target_log_prob_fn = joint_dist_pinned.unnormalized_log_prob

    if seed is None:
        seed = 26401
    run_mcmc_seed = samplers.sanitize_seed(seed, salt="run_mcmc_seed")

    if init_state is None:
        if pins:
            init_state_ = joint_dist_pinned.sample_unpinned(n_chains)
        else:
            init_state_ = joint_dist_pinned.sample(n_chains)
        ini_state_unbound = bijector.inverse(init_state_)
        run_mcmc_seed, *init_seed = samplers.split_seed(
            run_mcmc_seed, n=len(ini_state_unbound) + 1
        )
        init_state = bijector.forward(
            tf.nest.map_structure(
                lambda x, seed: tfd.Uniform(-1.0, tf.constant(1.0, x.dtype)).sample(
                    x.shape, seed=seed
                ),
                ini_state_unbound,
                tf.nest.pack_sequence_as(ini_state_unbound, init_seed),
            )
        )

    @tf.function(autograph=False, jit_compile=True)
    def run_inference_nuts(init_state, draws, tune, seed):
        seed, tuning_seed, sample_seed = samplers.split_seed(seed, n=3)

        def gen_kernel(step_size):
            hmc = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=target_log_prob_fn, step_size=step_size
            )
            hmc = tfp.mcmc.TransformedTransitionKernel(hmc, bijector=bijector)
            tuning_hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
                hmc, tune // 2, target_accept_prob=0.85
            )
            return tuning_hmc

        def tuning_trace_fn(_, pkr):
            return pkr.inner_results.transformed_state, pkr.new_step_size

        def get_tuned_stepsize(samples, step_size):
            return tf.math.reduce_std(samples, axis=0) * step_size[-1]

        step_size = tf.nest.map_structure(tf.ones_like, bijector.inverse(init_state))
        tuning_hmc = gen_kernel(step_size)
        init_samples, (sample_unbounded, tuning_step_size) = tfp.mcmc.sample_chain(
            num_results=200,
            num_burnin_steps=tune // 2 - 200,
            current_state=init_state,
            kernel=tuning_hmc,
            trace_fn=tuning_trace_fn,
            seed=tuning_seed,
        )

        tuning_step_size = tf.nest.pack_sequence_as(sample_unbounded, tuning_step_size)
        step_size_new = tf.nest.map_structure(
            get_tuned_stepsize, sample_unbounded, tuning_step_size
        )
        sample_hmc = gen_kernel(step_size_new)

        def sample_trace_fn(_, pkr):
            energy_diff = unnest.get_innermost(pkr, "log_accept_ratio")
            return {
                "target_log_prob": unnest.get_innermost(pkr, "target_log_prob"),
                "n_steps": unnest.get_innermost(pkr, "leapfrogs_taken"),
                "diverging": unnest.get_innermost(pkr, "has_divergence"),
                "energy": unnest.get_innermost(pkr, "energy"),
                "accept_ratio": tf.minimum(1.0, tf.exp(energy_diff)),
                "reach_max_depth": unnest.get_innermost(pkr, "reach_max_depth"),
            }

        current_state = tf.nest.map_structure(lambda x: x[-1], init_samples)
        return tfp.mcmc.sample_chain(
            num_results=draws,
            num_burnin_steps=tune // 2,
            current_state=current_state,
            kernel=sample_hmc,
            trace_fn=sample_trace_fn,
            seed=sample_seed,
        )

    mcmc_samples, mcmc_diagnostic = run_inference_nuts(
        init_state, n_draws, num_adaptation_steps, run_mcmc_seed
    )

    if return_compiled_function:
        return mcmc_samples, mcmc_diagnostic, run_inference_nuts
    else:
        return mcmc_samples, mcmc_diagnostic


def train_variables(
    loss_fn,
    trainable_variables,
    epochs,
    optimizer=keras.optimizers.Adam(),
    callbacks=None,
):
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = loss_fn()
            grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))


def MultivariateNormalLKJ(mu, rho_chol, scale_diag, name):
    scale_tril = tf.linalg.LinearOperatorDiag(scale_diag) @ rho_chol
    return tfd.MultivariateNormalTriL(mu, scale_tril, name=name)


class LowRankNormal(tfd.Distribution):
    def __init__(self, b, c, name="LowRankNormal"):
        self._reparameterization_type = tfd.Normal(0, 1).reparameterization_type
        self.b = b
        self.c = c

    def _batch_shape(self):
        return self.c.shape[:-1]

    def _event_shape(self):
        return self.c.shape[-1:]

    def sample(self, sample_shape=(), seed=None, name="a"):
        d = self.b.shape[-2]
        f = self.b.shape[-1]
        means = tf.concat(
            [tf.zeros_like(self.c), tf.zeros_like(self.c[..., :f])], axis=-1
        )
        s = tfd.Normal(means, 1).sample(sample_shape, seed=seed)
        return s[..., :d] * self.c + tf.einsum("...f,...df->...d", s[..., d:], self.b)

    def log_prob(self, x):
        b = self.b
        c = self.c
        f = self.b.shape[-1]
        d = self.b.shape[-2]
        c2inv = 1 / c**2
        btC2invB = tf.einsum("...df,...d,...dF->...fF", self.b, c2inv, self.b)
        det = tf.reduce_prod(c**2, axis=-1) * tf.linalg.det(tf.eye(f) + btC2invB)
        Q1 = tf.linalg.inv(tf.eye(f) + btC2invB)
        Σinv = tf.linalg.diag(c2inv) - tf.einsum(
            "...i,...im,...mk,...jk,...j->...ij", c2inv, self.b, Q1, self.b, c2inv
        )
        return (
            -x.shape[-1] / 2 * tf.math.log(2 * PI)
            - 0.5 * tf.math.log(det)
            - 0.5 * tf.einsum("nd,...dD,nD->...n", x, Σinv, x)
        )


class DiscreteNormal(tfd.Distribution):
    def __init__(self, loc, scale, name):
        self.dist = tfd.Normal(loc, scale)
        self._name = name
        self._reparameterization_type = self.dist.reparameterization_type
        self._dtype = loc.dtype if hasattr(loc, "dtype") else tf.constant(loc).dtype

    def _batch_shape(self):
        return self.dist.batch_shape

    def _event_shape(self):
        return self.dist.event_shape

    def log_prob(self, y):
        return tf.math.log(self.dist.cdf(y + 1) - self.dist.cdf(y))

    def sample(self, *args, **kwargs):
        return tf.math.floor(self.dist.sample(*args, **kwargs))


class AutonormalGuide(keras.layers.Layer):
    def __init__(
        self,
        target,
        event_shape=None,
        bijectors="experimental_default_event_space_bijector",
        init_scale=1,
        operators="diag",
        **kwargs,
    ):
        """
        :param target:
        :param init_scale:
        :param operators: Example operators for a model with 2 block variables representing full covariance
            operators = (
                (tf.linalg.LinearOperatorLowerTriangular,),
                (tf.linalg.LinearOperatorFullMatrix, tf.linalg.LinearOperatorLowerTriangular),
            )
        :return:
        """
        super().__init__(**kwargs)
        event_shape = target.event_shape if event_shape is None else event_shape
        event_sizes = get_event_size(event_shape)
        baseDistribution = tfd.JointDistributionSequential(
            [tfd.Sample(tfd.Normal(0, init_scale), x) for x in event_sizes]
        )
        scale_bijector = get_block_scale_bijector(event_shape, operators)

        shift_bijector = get_loc_bijector(event_shape)

        event_space_bijector = []
        if (
            bijectors == "experimental_default_event_space_bijector"
            and target is not None
        ):
            event_space_bijector = [
                target.experimental_default_event_space_bijector(),
                tfb.Restructure(
                    tf.nest.pack_sequence_as(event_shape, range(len(event_shape)))
                ),
            ]
        elif bijectors is not None:
            event_space_bijector = [
                tfb.Restructure(
                    tf.nest.pack_sequence_as(event_shape, range(len(event_shape)))
                ),
                tfb.JointMap(bijectors),
            ]

        bijector = tfb.Chain(
            [
                *event_space_bijector,
                tfb.JointMap(
                    [
                        tfb.Reshape(s)
                        for s in [tf.constant(x, dtype="int32") for x in event_shape]
                    ]
                ),
                shift_bijector,
                scale_bijector,
            ]
        )
        surrogatePosterior = bijector(baseDistribution)
        self.guide = surrogatePosterior

    def __call__(self, data=None):
        return self.guide


def cross_product(*a):
    """
    :param a: a ... x (k-1) x k tensor where k is the dimensino
    :return: a ... x k matrix which is orthogonal to the ... x k-1 vectors
    """
    if len(a) > 1:
        a = tf.concat([tf.constant(x)[..., None, :] for x in a], axis=-2)
    else:
        a = a[0]
    k = a.shape[-1]
    idx = set(range(k))
    out = []
    for i in range(k):
        out.append(list(idx - {i}))
    idx = tf.stack(out)  # .reshape(-1)
    x = tf.gather(a.T, idx)
    c = tf.pow(-1.0, tf.range(k, dtype=a.dtype))
    return tf.linalg.det(x) * c


def concat_shapes(shapes):
    return tf.concat(list(shapes), axis=0)


def broadcast_to(y, shapes):
    return tf.broadcast_to(y, concat_shapes(shapes))


class StickBreaking(tfb.Bijector):
    def __init__(self, validate_args=False, name="StickBreaking"):
        super(StickBreaking, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=1, name=name
        )

    def _forward(self, x):
        o = tf.ones_like(x[..., :1])
        return x * tf.concat([o, tf.math.cumprod(1 - x[..., :-1], axis=-1)], axis=-1)

    def _inverse(self, y):
        cumsum = tf.concat(
            [tf.ones_like(y[..., :1]), 1 - tf.math.cumsum(y[..., :-1], axis=-1)],
            axis=-1,
        )
        return y / cumsum

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self._inverse(y))

    def _forward_log_det_jacobian(self, x):
        return tf.reduce_sum(tf.cumsum(tf.math.log(1 - x), axis=-1), axis=-1)


az.data.inference_data.InferenceData.dict = property(asdict)
