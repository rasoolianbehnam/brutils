import os
v = int(os.environ.get('pymc_version', '4'))
if v == 3:
    import pymc3 as pm
    from pymc3.distributions import Interpolated
    from theano import shared, tensor as tt
else:
    import pymc as pm
    from pymc.distributions import Interpolated
    from aesara import shared, tensor as tt

from functools import partial, wraps
from empiricaldist import Pmf
from IPython.core.magic import register_cell_magic, needs_local_scope
from scipy.stats import gaussian_kde
import numpy as np
from scipy import stats
import arviz as az

from brutils.utility import RegisterWithClass


def hierarchical_mvNormal_old(name, sd_dist, dims, mu=0.0, eta=2.0):
    chol, corr, stds = pm.LKJCholeskyCov(f"chol_{name}", n=len(dims), eta=eta, sd_dist=sd_dist, compute_corr=True)
    z = pm.Normal(f"z_{name}", mu, 1.0, dims=dims[::-1])
    out = pm.Deterministic(name, tt.dot(chol, z).T, dims=dims)
    return out


def hierarchical_mvNormal(name, z, sd_dist, dims, eta=2.0):
    """
    β = pm.Normal('β', -1, .5, dims="caffe")
    α = pm.Normal('α', 5, 2, dims="caffe")
    z = pm.Deterministic('z', tt.stack([α, β]), dims=("parameter", "caffe"))
    μs = pms.hierarchical_mvNormal('μs', z,
                                   pm.Exponential.dist(1),
                                   dims=("caffe", "parameter"),
                                  )
    """
    n = int(z.shape.get_test_value()[0])
    chol, corr, stds = pm.LKJCholeskyCov(f"chol_{name}", n=n,
                                         eta=eta, sd_dist=sd_dist, compute_corr=True)
    out = pm.Deterministic(name, tt.dot(chol, z).T, dims=dims)
    return out


def hierarchical_normal(name, dims=None, sigma=5, shape=None, μ=0., sigma_dist=pm.HalfCauchy):
    out1, out2 = get_sigma_and_delta(name, sigma, μ, sigma_dist, dims=dims, shape=shape)
    return out1


def get_sigma_and_delta(name, sigma, μ, sigma_dist, dims=None, shape=None):
    σ = sigma_dist(f"σ_{name}", sigma)
    Δ = pm.Normal(f"μ_{name}", 0, 1, dims=dims, shape=shape)
    return pm.Deterministic(name, μ + Δ * σ, dims=dims), pm.Deterministic(name + "_single", μ + Δ * σ)


def set_data_values(data, df):
    df = df.obj_to_code()
    keys = set(data._asdict().keys())
    assert set(df).issubset(keys)
    for key in set(df):
        d = getattr(data, key)
        d.set_value(df[key].values.astype(d.dtype))


def beta_from_mean_dispersion(pbar, theta):
    return pbar * theta, (1.0 - pbar) * theta


def from_posterior_(param, samples, qs):
    """Make a kernel density estimate from a sample."""
    kde = gaussian_kde(samples)
    ps = kde(qs)
    return Interpolated(param, qs, ps)


def from_posterior(param, samples):
    smin, smax = np.min(samples) - np.std(samples), np.max(samples) + np.std(samples)
    qs = np.linspace(smin, smax, 81)
    return from_posterior_(param, samples, qs)


def kde_from_sample_(sample, qs):
    """Make a kernel density estimate from a sample."""
    kde = gaussian_kde(sample)
    ps = kde(qs)
    pmf = Pmf(ps, qs)
    pmf.normalize()
    return pmf


def kde_from_sample(sample, qs=None, r=.3):
    """Make a kernel density estimate from a sample."""
    if qs is None:
        smin, smax = np.min(sample) - np.std(sample) * r, np.max(sample) + np.std(sample) * r
        qs = np.linspace(smin, smax, 81)
    return kde_from_sample_(sample, qs)


@RegisterWithClass(pm.backends.base.MultiTrace)
def idata(self, *args, **kwargs):
    return az.from_pymc3(self, *args, **kwargs)


@RegisterWithClass(Pmf)
def Update(self, data):
    likelihood = self.likelihood(self.qs, data)
    out = self * likelihood
    out.normalize()
    return out


@RegisterWithClass(Pmf)
def UpdateSet(self, datas):
    out = np.array(self)
    for data in datas:
        likelihood = self.likelihood(self.qs, data)
        out *= likelihood
    out = Pmf(out, self.qs)
    out.normalize()
    return out


@RegisterWithClass(Pmf)
def LogUpdateSet(self, datas):
    out = np.log(self)
    for data in datas:
        likelihood = np.log(self.likelihood(self.qs, data))
        out += likelihood
    out -= out.min()
    out = Pmf(np.exp(out), self.qs)
    out.normalize()
    return out


@RegisterWithClass(az.InferenceData)
def add_data(self, post=None, prior=None, log_likelihood=None, *args, **kwargs):
    self.add_groups({'posterior_predictive': post, 'prior': prior, 'log_likelihodd': log_likelihood}, *args, **kwargs)


@RegisterWithClass(az.InferenceData)
def add_post(self, *args, **kwargs):
    self.add_groups({'posterior_predictive': pm.sample_posterior_predictive(self, *args, **kwargs)})


pm.Model.var_dict = property(lambda self: {x.name: x for x in self.vars})

plot_forest = partial(az.plot_forest, combined=True, r_hat=True)
sample = partial(pm.sample, return_inferencedata=True)
idata = az.from_pymc3
