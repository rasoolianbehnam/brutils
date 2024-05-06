import numpy as np
import tensorflow as tf

keras = tf.keras
kl = keras.layers
einsum = tf.einsum


class Embeddings(kl.Layer):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = kl.Embedding(config.num_pools, config.hidden_size, mask_zero=True)
        self.layer_norm = kl.LayerNormalization(epsilon=1e-12)
        self.dropout = kl.Dropout(config.embedding_dropout_rate)

        def __call__(self, input_ids):
            # Create position IDs for input sequence
            embeddings = self.token_embeddings(input_ids)
            return embeddings
            embeddings = self.layer_norm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings


def scaled_dot_product_attention(query, key, value, reaches, masked):
    dim_k = np.int32(query.shape[-1])
    scores = einsum("...td,...Td->...tT", query, key) / np.sqrt(dim_k)
    mask = einsum("...td,...Td->...tT", reaches, reaches)
    scores = tf.where(mask > 0, scores, -np.inf)
    if masked:
        scores = tf.where(
            tf.linalg.band_part(tf.ones_like(scores, dtype='int32'), -1, 0) == 0,
            -1e10, scores
        )
    scores = tf.math.softmax(scores, axis=-1)
    scores = tf.where(tf.math.is_nan(scores), 0., scores)
    return einsum("...tT,...Td->...td", scores, value) * reaches


class AttentionHead(kl.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.q = keras.Sequential([
            kl.Dense(config.hidden_size, activation='relu'),
            kl.Dropout(config.hidden_dropout_prob),
            kl.Dense(config.hidden_size),
        ])
        self.k = keras.Sequential([
            kl.Dense(config.hidden_size, activation='relu'),
            kl.Dropout(config.hidden_dropout_prob),
            kl.Dense(config.hidden_size),
        ])

        self.v = keras.Sequential([
            kl.Dense(config.hidden_size, activation='relu'),
            kl.Dropout(config.hidden_dropout_prob),
            kl.Dense(config.hidden_size),
        ])

        self.dense_reach = keras.Sequential([
            kl.Dense(config.reach_embedding_size // 2),
            kl.Dense(config.reach_embedding_size),
        ])
        self.masked = config.masked

    def __call__(self, hidden_state, reaches):
        reaches_transformed = self.dense_reach(reaches[..., None])
        hidden_with_reach = tf.concat([hidden_state, reaches_transformed], axis=-1)  # * sequence_mask
        k = self.k(hidden_with_reach)  # * sequence_mask
        q = self.q(hidden_with_reach)  # * sequence_mask
        v = self.v(hidden_with_reach)  # * sequence_mask
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_with_reach),
            self.k(hidden_with_reach),
            v, reaches[..., None], masked=self.masked
        )
        return attn_outputs


class Reach(kl.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.transformer = AttentionHead(config)
        self.dense01 = keras.Sequential([
            kl.Dense(config.hidden_size // 2, activation='relu'),
            kl.Dropout(config.reach_dropout_prob),
            kl.Dense(1, activation='sigmoid'),
        ])

    def __call__(self, x, r):
        n = r.shape[-1]
        r_max = tf.reduce_max(r, axis=-1, keepdims=True)
        r_sum = tf.reduce_sum(r, axis=-1, keepdims=True)

        correction_factor = tf.where(r_sum > 1, (1 - r_max) / (r_sum - r_max), 1.)

        z = self.transformer(x, r)
        ratio = self.dense01(z)[..., 0] * correction_factor
        ratio = tf.where(r > 0, ratio, 0.)

        ratio = tf.where(tf.greater_equal(r, r_max), 1., ratio)
        return einsum("...t,...t->...", ratio, r)


# x, r = tf.random.uniform([8, 9, 10]), tf.random.uniform([8, 9])/5
# Reach(config)(x, r)

# Reach(config)(sample_e, sample_r)

class TransformerEncoder(kl.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = Embeddings(config)
        self.reach = Reach(config)

    def __call__(self, input_ids, r):
        x = self.embeddings(input_ids)
        return self.reach(x, r)


# TransformerEncoder(config)(val_titles[:8], val_reaches[:8])

class TransformerModel(keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.m = TransformerEncoder(config)

    def call(self, xr):
        x, r = xr
        return self.m(x, r)
