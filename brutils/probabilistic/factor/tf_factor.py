import itertools
import operator
from functools import partial, reduce

import numpy as np
from collections import OrderedDict
import xarray as xr
import tensorflow as tf
MAXINT = np.iinfo('int').max


def __align_ndarrays__(a1, v1, v2):
    all_vars = sorted(set(v1 + v2))
    v1_idx = {v: k for k, v in enumerate(v1)}
    b1 = a1
    for i in set(all_vars) - set(v1):
        b1 = tf.expand_dims(b1, -1)

    k = len(a1.shape)
    tmp = []
    for i in all_vars:
        t = v1_idx.get(i, None)
        if t is None:
            t = k
            k += 1
        tmp.append(t)
    return tf.transpose(b1, tmp), all_vars


def list_to_ordered_dict(variables):
    d = OrderedDict()
    for k, v in enumerate(variables):
        d[v] = k
    return d


class TfFactor:
    def __init__(self, variables, *, domains=None, value=None, default=0):
        self.variables = list_to_ordered_dict(variables)
        # if not isinstance(value, tf.Variable):
        #     value = tf.Variable(value)
        self.value = value
        if value is None:
            assert domains is not None
            self.value = tf.ones(domains) * default

    @property
    def vars(self):
        return list(self.variables)

    @property
    def domains(self):
        return list(self.value.shape)

    @property
    def shapes(self):
        return list(self.value.shape)

    def __eq__(self, other):
        if set(self.vars) != set(other.vars):
            return False
        return np.allclose(
            self.rearrange_variables(other.variables).value,
            other.value
        )

    def sort_variables(self):
        new_variables = sorted(self.variables)
        return self.rearrange_variables(new_variables)

    def rearrange_variables(self, new_variables):
        new_value = tf.transpose(
            self.value,
            [self.variables[i] for i in new_variables]
        )
        return TfFactor(new_variables, value=new_value)

    def align_with(self, other_factor):
        value, variables = __align_ndarrays__(
            self.value, list(self.variables),
            list(other_factor.variables)
        )
        return TfFactor(variables, value=value)

    def copy(self):
        return TfFactor([v for v in self.variables], value=self.value * 1)

    def __matmul__(self, other):
        new_variables = list(set(self.vars + other.vars))
        return TfFactor(new_variables, value=tf.zeros([1] * len(new_variables)))

    def __mul__(self, other):
        if len(self.variables) == 0:
            return other.copy()
        if len(other.variables) == 0:
            return self.copy()
        a = self.align_with(other)
        b = other.align_with(self)
        a.value = tf.abs(a.value * b.value)
        return a

    def __add__(self, other):
        a = self.align_with(other)
        b = other.align_with(self)
        a.value = tf.abs(a.value) + tf.abs(b.value)
        return a

    def __sub__(self, other):
        a = self.align_with(other)
        b = other.align_with(self)
        a.value = a.value - b.value
        return a

    def __truediv__(self, other):
        if not isinstance(other, TfFactor):
            return TfFactor(self.variables, value=self.value / other)
        a = self.align_with(other)
        b = other.align_with(self)
        a.value = tf.abs(a.value) / tf.abs(b.value)
        return a

    def marginalise(self, vars_to_marginalise):
        new_variables = list_to_ordered_dict(
            [x for x in self.variables
             if x not in vars_to_marginalise]
        )
        new_value = tf.reduce_sum(
            tf.abs(self.value),
            axis=[self.variables[i] for i in vars_to_marginalise]
        )
        return TfFactor(new_variables, value=new_value)

    def dummy_marginalise(self, vars_to_marginalise):
        new_value = self.value[[0 if x in vars_to_marginalise else slice(None) for x in self.vars]]
        new_vars = [x for x in self.vars if x not in vars_to_marginalise]
        return TfFactor(new_vars, value=new_value)

    def max_marginalise(self, vars_to_marginalise):
        new_variables = list_to_ordered_dict(
            [x for x in self.variables
             if x not in vars_to_marginalise]
        )
        new_value = tf.reduce_max(
            self.value,
            axis=[self.variables[i] for i in vars_to_marginalise]
        )
        return TfFactor(new_variables, value=new_value)

    def marginalise_in(self, vars_to_keep):
        return self.marginalise(set(self.variables) - set(vars_to_keep))

    def max_marginalise_in(self, vars_to_keep):
        return self.max_marginalise(set(self.variables) - set(vars_to_keep))

    def log_transform(self):
        return TfFactor(tf.abs(self.variables), value=tf.math.log(self.value))

    def normalize(self):
        v = tf.abs(self.value)
        return TfFactor(
            self.variables,
            value=v / tf.reduce_sum(v)
        )

    def conditional_normalize(self, given):
        marginalized = self.marginalise(given)
        return self / marginalized

    @property
    def Z(self):
        return tf.reduce_sum(self.value)

    def evidence(self, evidence):
        idx0 = [(evidence.get(k, None)) for k in self.variables]
        idx = [slice(x) if x is None else x for x in idx0]
        new_vars = [v1 for v1, v2 in zip(self.variables, idx0) if v2 is None]
        return TfFactor(variables=new_vars, value=self.value[idx])

    def to_xarray(self):
        from brutils.misc.xr_factor import XrFactor
        return XrFactor(value=xr.DataArray(self.value.numpy(), dims=list(str(v) for v in self.variables)))

    @staticmethod
    def from_xarray(factor, var_to_int=False):
        vrs = factor.vars
        if var_to_int:
            vrs = [int(v) for v in factor.vars]
        return TfFactor(vrs, value=factor.value.values)

    @staticmethod
    def from_factor(factor, var_to_int=True):
        from brutils.misc.xr_factor import XrFactor
        return TfFactor.from_xarray(XrFactor.from_factor(factor), var_to_int)

    @staticmethod
    def from_matlab(factor, var_to_int=True):
        from brutils.misc.factor import Factor
        return TfFactor.from_factor(Factor.from_matlab(factor), var_to_int)

    @property
    def df(self):
        return self.to_xarray().value.to_series().rename('value').reset_index()

    @property
    def isjpd(self):
        return np.allclose(self.Z, 1)

    @property
    def isconditional(self):
        return len(self.given) > 0

    @property
    def given(self):
        d = list(zip(
            itertools.combinations(self.vars, len(self.vars)-1),
            itertools.combinations(self.shapes, len(self.vars)-1)
        ))
        z = self.Z
        return list(itertools.chain.from_iterable([k for k, v in d if np.allclose(np.prod(v), z, atol=1e-2)]))

    @property
    def conditional(self):
        return list(set(self.vars)-set(self.given))


TfFactor.marginalize = TfFactor.marginalise
TfFactor.marginalize_in = TfFactor.marginalise_in
TfFactor.max_marginalize = TfFactor.max_marginalise
TfFactor.max_marginalize_in = TfFactor.max_marginalise_in


def ipf(*conditionals, weights=None, init=None, max_iter=MAXINT, learning_rate=1e-5):
    if weights is None:
        weights = np.ones(len(conditionals))
    given = conditionals[0].conditional
    if init is None:
        sol = reduce(operator.mul, conditionals).normalize()
        sol.value = np.random.rand(*sol.value.shape)
        sol.value = tf.Variable(sol.value/tf.reduce_sum(sol.value), name="mardas")
    else:
        sol = init
    best_sol = tf.constant(sol.value)
    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    hist = [MAXINT]
    append_interval = 20
    hist_lookback = 30
    min_iter_for_lookback = append_interval * hist_lookback
    for i in range(max_iter):
        with tf.GradientTape() as tape:
            ss = []
            for idx, conditional in enumerate(conditionals):
                mardas = sol.marginalize_in(conditional.vars).conditional_normalize(given).sort_variables()
                blah = conditional.sort_variables()
                ss.append(tf.reduce_max(
                    tf.abs(
                        mardas.value - blah.value
                    )/blah.value
                )*weights[idx])
            s = sum(ss)
            grads = tape.gradient(s, [sol.value])
        if i % append_interval == 0:
            if s > min(hist):
                best_sol = tf.constant(sol.value)
            hist.append(s)
            if i > min_iter_for_lookback:
                if np.polyfit(np.arange(hist_lookback), hist[-hist_lookback:], 1)[0] > -1e-7:
                    break
        opt.apply_gradients(zip(grads, [sol.value]))
        if i % 500 == 0:
            print(s)
    return TfFactor(variables=sol.vars, value=best_sol).normalize()
