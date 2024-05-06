import itertools
import operator
from functools import partial, reduce

import numpy as np
import pandas as pd
from collections import OrderedDict
import xarray as xr


def list_to_ordered_dict(variables):
    d = OrderedDict()
    for k, v in enumerate(variables):
        d[v] = k
    return d


def is_iterable(x):
    try:
        iter(x)
        return True
    except:
        return False


def args_to_list(vars_to_marginalise):
    out = pd.Series(vars_to_marginalise).explode().list
    # out = list(itertools.chain.from_iterable(vars_to_marginalise))
    return out


class XrFactor:
    def __init__(self, *, variables=None, value=None, coords=None, default=0):
        if isinstance(value, xr.DataArray):
            self.value = value
            return
        if isinstance(value, pd.DataFrame):
            value_col = list(set(value) - set(variables))[0]
            self.value = xr.DataArray.from_series(value.set_index(variables)[value_col])
            return
        if coords is not None:
            value = value if value is not None else np.ones(list(map(len, coords.values()))) * default
            self.value = xr.DataArray(value, coords=coords)
        else:
            self.value = xr.DataArray([])

    def copy(self):
        return XrFactor(value=self.value)

    @property
    def values(self):
        return self.value.values

    @property
    def variables(self):
        return list(self.value.dims)

    @property
    def domains(self):
        return [self.value.coords.get(v).to_numpy() for v in self.variables]

    @property
    def shapes(self):
        return [len(x) for x in self.domains]

    def __repr__(self):
        return repr(self.value)

    @property
    def vars(self):
        return self.variables

    def __eq__(self, other):
        if set(self.vars) != set(other.vars):
            return False
        return np.allclose(self.value.transpose(*other.variables), other.value)

    def __mul__(self, other):
        return XrFactor(value=self.value * other.value)

    def __truediv__(self, other):
        return XrFactor(value=self.value / other.value)

    def __add__(self, other):
        return XrFactor(value=self.value + other.value)

    def __sub__(self, other):
        return XrFactor(value=self.value - other.value)

    def marginalise(self, *vars_to_marginalise):
        if len(vars_to_marginalise) == 0:
            return self.copy()
        vars_to_marginalise = args_to_list(vars_to_marginalise)
        return XrFactor(value=self.value.sum(vars_to_marginalise))

    def max_marginalise(self, vars_to_marginalise):
        return XrFactor(value=self.value.max(vars_to_marginalise))

    def log_transform(self):
        return XrFactor(value=np.log(self.value))

    def marginalise_in(self, *vars_to_keep):
        vars_to_keep = args_to_list(vars_to_keep)
        vars_to_remove = list(set(self.variables) - set(vars_to_keep))
        return self.marginalise(*vars_to_remove)

    def normalize(self):
        return XrFactor(value=self.value / self.value.sum())

    def conditional_normalize(self, *given):
        return XrFactor(value=self.value / self.value.sum(args_to_list(given)))

    @property
    def Z(self):
        return self.value.sum().values

    def evidence(self, evidence):
        evidence = {k: v for k, v in evidence.items() if k in self.vars}
        return XrFactor(value=self.value.sel(**evidence))

    def to_tf_factor(self):
        from brutils.misc.tf_factor import TfFactor
        return TfFactor(self.variables, value=self.value.values)

    def to_df(self):
        return self.value.to_dataframe('p').reset_index()

    @staticmethod
    def from_factor(factor):
        new_vars = [str(v) for v in factor.vars]
        s = factor.df.rename(columns=dict(zip(factor.vars, new_vars))).set_index(new_vars).value
        value = xr.DataArray.from_series(s)
        return XrFactor(value=value)

    @property
    def isjpd(self):
        return np.allclose(self.Z, 1, atol=1e-2)

    @property
    def isconditional(self):
        return len(self.given) > 0

    @property
    def given(self):
        if self.isjpd:
            return self.vars
        if self.isjpd:
            return []
        a = list(itertools.combinations(self.vars, len(self.vars) - 1))
        b = list(itertools.combinations(self.shapes, len(self.vars) - 1))
        stds = [self.marginalise(x).value.std().values for x in a]
        return list(a[np.argmax(stds)])

    @property
    def conditional(self):
        return list(set(self.vars) - set(self.given))

    def mutate(self, c=1e-15):
        r = np.random.rand(*self.value.shape)*c
        return XrFactor(value=self.value+r, variables=self.vars)

    def display_cpd(self):
        return self.value.to_dataframe('p').reset_index().set_index(self.given+self.conditional).unstack().p



XrFactor.marginalize = XrFactor.marginalise
XrFactor.marginalize_in = XrFactor.marginalise_in


def ipf(*marginals, init="random", max_iter=1000, tol=1e-10):
    # variables = list(reduce(operator.or_, [set(m.vars) for m in marginals]))
    variables = list(itertools.chain.from_iterable([m.vars for m in marginals]))
    shapes = list(itertools.chain.from_iterable([m.shapes for m in marginals]))
    domains = list(itertools.chain.from_iterable([m.domains for m in marginals]))
    vars_and_shapes = dict(zip(variables, shapes))
    vars_and_domains = dict(zip(variables, domains))
    if init == "random":
        sol = XrFactor(
            value=xr.DataArray(np.random.rand(*vars_and_shapes.values()), coords=vars_and_domains)
        )
    else:
        sol = reduce(operator.mul, marginals)
    i = 0
    for i in range(max_iter):
        max_ratio = -10
        for marginal in marginals:
            ratio = marginal.value / sol.marginalise_in(marginal.vars).value
            max_ratio = np.max([max_ratio, np.max(ratio)])
            sol.value = sol.value * ratio
        if np.abs(max_ratio - 1) < tol:
            break
    if i == max_iter - 1:
        print(f"max iteration reach before tolerance. max tolerance is {max_ratio}")
    return sol

