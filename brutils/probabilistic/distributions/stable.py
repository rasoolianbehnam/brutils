import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
from brutils.imports.implicits.tensorflow import broadcast_to, Ind, Root

tfd = tfp.distributions
mth = tf.math
inf = tf.constant(np.float32('inf'))

RADIUS = .01


def _unsafe_standard_stable(alpha, beta, V, W, coords):
    # Implements a noisily reparametrized version of the sampler
    # Chambers-Mallows-Stuck method as corrected by Weron [1,3] and simplified
    # by Nolan [4]. This will fail if alpha is close to 1.

    # Differentiably transform noise via parameters.
    # assert V.shape == W.shape
    inv_alpha = mth.reciprocal(alpha)
    half_pi = tf.constant(math.pi / 2, dtype='float32')
    eps = tf.keras.backend.epsilon()
    # make V belong to the open interval (-pi/2, pi/2)
    V = tf.clip_by_value(V, 2 * eps - half_pi, half_pi - 2 * eps)
    ha = half_pi * alpha
    b = beta * mth.tan(ha)
    # +/- `ha` term to keep the precision of alpha * (V + half_pi) when V ~ -half_pi
    v = mth.atan(b) - ha + alpha * (V + half_pi)
    Z = (
            mth.sin(v)
            / mth.pow(mth.reciprocal(mth.sqrt(1 + b * b)) * mth.cos(V), inv_alpha)
            * mth.pow(tf.clip_by_value(mth.cos(v - V), eps, inf) / W, inv_alpha - 1)
    )
    Z = tf.where(Z != Z, 0., Z)

    # Optionally convert to Nolan's parametrization S^0 where samples depend
    # continuously on (alpha,beta), allowing interpolation around the hole at
    # alpha=1.
    if coords == "S0":
        return Z - b
    elif coords == "S":
        return Z
    else:
        raise ValueError("Unknown coords: {}".format(coords))




def _standard_stable(alpha, beta, aux_uniform, aux_exponential, coords):
    """
    Differentiably transform two random variables::

        aux_uniform ~ Uniform(-pi/2, pi/2)
        aux_exponential ~ Exponential(1)

    to a standard ``Stable(alpha, beta)`` random variable.
    """
    # Determine whether a hole workaround is needed.
    if coords == "S":
        # S coords are discontinuous, so interpolate instead in S0 coords.
        Z = _standard_stable(alpha, beta, aux_uniform, aux_exponential, "S0")
        return tf.where(alpha == 1., Z, Z + beta * mth.tan(math.pi / 2 * alpha))

    hole = 1.0
    near_hole = tf.stop_gradient(
        mth.abs(alpha - hole) <= RADIUS
    )
    unsafe = _unsafe_standard_stable(
        alpha, beta, aux_uniform, aux_exponential, coords=coords
    )

    # Avoid the hole at alpha=1 by interpolating between pairs
    # of points at hole-RADIUS and hole+RADIUS.
    aux_uniform_ = aux_uniform[..., None]
    aux_exponential_ = aux_exponential[..., None]
    beta_ = beta[..., None]
    alpha_ = broadcast_to(
        tf.constant([hole - RADIUS, hole + RADIUS]),
        [tf.shape(alpha), [2]]
    )

    def get_weights():
        # We don't need to backprop through weights, since we've pretended
        # alpha_ is reparametrized, even though we've clamped some values.
        #               |a - a'|
        # weight = 1 - ----------
        #              2 * RADIUS
        weights = -(mth.abs(alpha_ - alpha[..., None]) / (2 * RADIUS)) + 1
        return weights

    weights = tf.stop_gradient(get_weights())
    pairs = _unsafe_standard_stable(
        alpha_, beta_, aux_uniform_, aux_exponential_, coords=coords
    )
    safe = tf.reduce_sum(pairs * weights, axis=-1)
    return tf.where(near_hole, safe, unsafe)


def _unsafe_shift(a, skew, t_scale):
    # At a=1 the lhs has a root and the rhs has an asymptote.
    return (mth.sign(skew) * t_scale - skew) * mth.tan(math.pi / 2 * a)


def _safe_shift(a, skew, t_scale, skew_abs):
    radius = 0.005
    hole = 1.0
    near_hole = mth.abs(a - hole) <= radius
    unsafe = _unsafe_shift(a, skew, t_scale)

    # Avoid the hole at a=1 by interpolating between points on either side.
    a_ = broadcast_to(
        tf.constant([hole - RADIUS, hole + RADIUS]),
        [tf.shape(a), [2]]
    )

    def get_weights():
        # We don't need to backprop through weights, since we've pretended
        # alpha_ is reparametrized, even though we've clamped some values.
        #               |a - a'|
        # weight = 1 - ----------
        #              2 * RADIUS
        weights = -(mth.abs(a_ - a[..., None]) / (2 * RADIUS)) + 1
        return weights

    weights = get_weights()
    skew_ = skew[..., None]
    skew_abs_ = skew_abs[..., None]
    t_scale_ = mth.pow(skew_abs_, mth.reciprocal(a_))
    pairs = _unsafe_shift(a_, skew_, t_scale_)
    safe = tf.reduce_sum(pairs * weights, -1)
    return tf.where(near_hole, safe, unsafe)


def reparameterized_stable_loc_scale(stability, skew, loc, scale, root=False, name="rep_stable", event_dims=0):
    if root:
        rt = Root
    else:
        rt = lambda x: x
    half_pi = tf.constant(math.pi / 2, dtype='float32')
    eps = tf.keras.backend.epsilon()
    ones = tf.ones_like(stability)
    sample_shape = tf.shape(stability)
    pp = tf.ones(sample_shape) # broadcast_to(, [sample_shape])
    zu = yield rt(Ind(tfd.Uniform(0, pp), event_dims, name=f"zu_{name}"))
    tu = yield rt(Ind(tfd.Uniform(0, pp), event_dims, name=f"tu_{name}"))
    zu = zu * half_pi * 2 - half_pi
    tu = tu * half_pi * 2 - half_pi
    ze = yield rt(Ind(tfd.Exponential(ones), event_dims, name=f"ze_{name}"))
    te = yield rt(Ind(tfd.Exponential(ones), event_dims, name=f"te_{name}"))

    a = stability
    z = _unsafe_standard_stable(a / 2, 1, zu, ze, coords="S")
    t = _standard_stable(a, ones, tu, te, coords="S0")
    a_inv = mth.reciprocal(a)

    skew_abs = tf.clip_by_value(mth.abs(skew), eps, 1 - eps)
    t_scale = mth.pow(skew_abs, a_inv)
    s_scale = mth.pow(1 - skew_abs, a_inv)
    shift = _safe_shift(a, skew, t_scale, skew_abs)
    loc = loc + scale * (mth.sign(skew) * t_scale * t + shift)
    scale = scale * s_scale * mth.sqrt(z) * mth.pow(mth.cos(math.pi / 4 * a), a_inv)
    scale = tf.clip_by_value(scale, eps, inf)
    return loc, scale


def reparameterized_stable(stability, skew, loc, scale, root=False, name="rep_stable", event_dims=0):
    loc, scale = yield from reparameterized_stable_loc_scale(stability, skew, loc, scale, root, name, event_dims)
    yield Ind(tfd.Normal(loc, scale), event_dims, name=name)
