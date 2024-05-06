from brutils.imports.implicits.tensorflow import *
import brutils.probabilistic.tfp_utils as tfpu


@tf.function
def forward_backward(init_probs, transition_probs, observation_dist, y):
    shape = tf.shape(transition_probs)
    batch_shape = shape[:-2]
    hidden_dim = shape[-1:]
    T = tf.shape(y)
    alpha = tf.zeros(tfpu.concat_shapes([T, shape[:-1]]))  # .at[[0]].set(init_probs)
    yy = tfpu.broadcast_to(y, [tf.ones_like(batch_shape), [1], tf.shape(y)]).T
    lamda = observation_dist.prob(yy)

    def forward_one_step(t, alpha, prev):
        alpha_t = prev * lamda[t]  # ... * h
        z = tf.reduce_sum(alpha_t, axis=-1, keepdims=True)
        alpha = alpha.at[[t]].set(alpha_t / z)
        prev = einsum("...i,...ij->...j", alpha[t], transition_probs)
        return t + 1, alpha, prev

    _, alpha, _ = tf.while_loop(
        lambda i, *_: i < T[0],
        forward_one_step,
        (0, alpha, init_probs),
        maximum_iterations=T[0]
    )

    final_prb = tf.ones_like(init_probs) / tf.cast(hidden_dim, 'float32')
    beta = tf.zeros(tfpu.concat_shapes([T, shape[:-1]])).at[T - 1].set(final_prb)

    def backward_one_step(t, beta):
        beta_t = einsum("...ij,...j,...j->...i", transition_probs, lamda[t], beta[t + 1])
        z = tf.reduce_sum(beta_t, axis=-1, keepdims=True)
        beta = beta.at[[t]].set(beta_t / z)
        return t - 1, beta

    _, beta = tf.while_loop(
        lambda i, *_: i >= 0,
        backward_one_step, (T[0] - 2, beta),
        maximum_iterations=T[0]
    )

    gamma = alpha * beta
    gamma = gamma / tf.reduce_sum(gamma, axis=-1, keepdims=True)
    return alpha, beta, gamma


def viterbi(init_probs, transition_probs, observation_dist, y):
    shape = tf.shape(transition_probs)
    batch_shape = shape[:-2]
    hidden_dim = shape[-1:]
    T = tf.shape(y)
    theta = tf.zeros(tfpu.concat_shapes([T, shape[:-1]])).at[T - 1].set(init_probs)
    a = tf.zeros(tfpu.concat_shapes([T, shape[:-1]]), dtype='int32')
    yy = tfpu.broadcast_to(y, [tf.ones_like(batch_shape), [1], tf.shape(y)]).T
    lamda = observation_dist.prob(yy)

    def forward_one_step(t, theta, a):
        theta_t = lamda[t] * tf.reduce_max(theta[t - 1, ..., None] * transition_probs, axis=-2)
        z = tf.reduce_sum(theta_t, axis=-1, keepdims=True)
        theta = theta.at[[t]].set(theta_t / z)
        a = a.at[[t]].set(tf.argmax(theta[t - 1, ..., None] * transition_probs, axis=-2))
        return t + 1, theta, a

    _, theta, a = tf.while_loop(
        lambda i, *_: i < T[0],
        forward_one_step,
        (0, theta, a),
        maximum_iterations=T[0]
    )

    z = tf.zeros(tfpu.concat_shapes([T, shape[:-2]]), dtype='int32').at[T - 1].set(tf.reduce_max(theta[-1], axis=-1))

    def backward_one_step(t, z):
        update = tf.gather(a[t + 1], z[t + 1], batch_dims=1)
        return t - 1, z.at[[t]].set(update)

    _, z = tf.while_loop(
        lambda i, *_: i >= 0,
        backward_one_step,
        (T[0] - 2, z),
        maximum_iterations=T[0]
    )

    return z
