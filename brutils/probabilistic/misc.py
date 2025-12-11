import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np


def break_vector(v, layer_sizes):
    s = 0
    for i, j in zip(layer_sizes[:-1], layer_sizes[1:]):
        idx = (i + 1) * j
        yield v[..., s : s + idx].reshape([-1, i + 1, j])
        s += idx


class HierarchicalDense(tf.keras.Layer):
    def __init__(
        self,
        input_dim,
        prior_input_dim,
        layer_sizes,
        n_data,
        activation="relu",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.layer_sizes = layer_sizes

        self.input_dim = input_dim
        self.prior_input_dim = prior_input_dim
        self.activation = tf.keras.activations.get(activation)
        self.n_data = n_data

    def build(self, shape):
        layer_sizes = [shape[-1]] + self.layer_sizes
        self.output_dim = np.prod(np.array(layer_sizes) + 1)
        self.output_dim = sum(
            [(i + 1) * j for i, j in zip(layer_sizes[:-1], layer_sizes[1:])]
        )
        self.embedding_layer = tf.keras.layers.Embedding(
            self.input_dim, self.output_dim * 2
        )
        self.prior_embedding = tf.keras.layers.Embedding(
            self.prior_input_dim, self.output_dim * 2
        )

    def get_total_loss(self, prior_indices):
        prior_params = self.prior_embedding(prior_indices)
        prior = tfd.Normal(
            prior_params[..., : self.output_dim],
            tf.math.exp(prior_params[..., self.output_dim :]),
        )
        embedding_params = self.embedding_layer(tf.range(self.input_dim))
        posterior = tfd.Normal(
            embedding_params[..., : self.output_dim],
            tf.math.exp(embedding_params[..., self.output_dim :]),
        )
        return tf.reduce_sum(posterior.kl_divergence(prior))

    def get_loss(self, I, PI):
        batch_size = tf.cast(tf.shape(I)[0], "float32")
        prior_params = self.prior_embedding(PI)
        prior = tfd.Normal(
            prior_params[..., : self.output_dim],
            tf.math.exp(prior_params[..., self.output_dim :]),
        )
        embedding_params = self.embedding_layer(I)
        posterior = tfd.Normal(
            embedding_params[..., : self.output_dim],
            tf.math.exp(embedding_params[..., self.output_dim :]),
        )
        return tf.reduce_sum(posterior.kl_divergence(prior)) / batch_size

    def dense(self, I, X):
        embedding_params = self.embedding_layer(I)
        embedding = tfd.Normal(
            embedding_params[..., : self.output_dim],
            tf.math.exp(embedding_params[..., self.output_dim :]),
        ).sample()
        for v in break_vector(embedding, self.layer_sizes):
            X = self.activation(
                tf.einsum("...dD,...d->...D", v[..., :-1, :], X) + v[..., -1, :]
            )
        return X

    def call(self, I, X, PI):
        self.add_loss(self.get_loss(I, PI))
        return self.dense(I, X)
