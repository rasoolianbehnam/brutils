from brutils.imports.implicits.tensorflow import *
import brutils.probabilistic.tfp_utils as tfpu
import tensorflow_probability as tfp

kl = keras.layers


def Combiner(z_dim, rnn_dim):
    z_t_1 = kl.Input([None, None, z_dim])
    h_rnn = kl.Input([None, rnn_dim])
    # out = tfp.layers.IndependentNormal(event_shape=z_dim)(locScale)
    lin_z_to_hidden = kl.Dense(rnn_dim, activation='tanh')
    h_combined = .5 * (lin_z_to_hidden(z_t_1) + h_rnn)
    lin_hidden_to_loc = kl.Dense(z_dim)
    lin_hidden_to_scale = kl.Dense(z_dim, activation='softplus')
    loc = lin_hidden_to_loc(h_combined)
    scale = lin_hidden_to_scale(h_combined)
    return keras.Model(inputs=[z_t_1, h_rnn], outputs=[loc, scale])


def GatedTransition(z_dim, transition_dim):
    a0 = kl.Input([None, None, z_dim])
    _gate = kl.Dense(transition_dim, activation='relu')(a0)
    gate = kl.Dense(z_dim, activation='sigmoid')(_gate)

    _proposed_mean = kl.Dense(transition_dim, activation='relu')(a0)
    proposed_mean = kl.Dense(z_dim)(_proposed_mean)

    lin_z_to_loc = kl.Dense(
        z_dim, 
        kernel_initializer=keras.initializers.identity(), 
        bias_initializer=keras.initializers.zeros()
    )
    loc = (1 - gate) * lin_z_to_loc(a0) + gate * proposed_mean
    scale = tf.math.softplus(kl.Dense(z_dim)(tf.nn.relu(proposed_mean)))
    # loc_scale = tf.stack([loc, scale], axis=-1)
    # out = tfp.layers.IndependentNormal()(loc_scale)
    return keras.Model(inputs=[a0], outputs=[loc, scale])


def Emitter(input_dim, z_dim, emission_dim):
    return keras.Sequential([
        kl.InputLayer([None, None, z_dim]),
        kl.Dense(emission_dim, activation='relu'),
        kl.Dense(emission_dim, activation='relu'),
        kl.Dense(input_dim, activation='sigmoid'),
        # tfp.layers.IndependentBernoulli(event_shape=input_dim)
        tfp.layers.DistributionLambda(lambda t: Ind(tfd.Bernoulli(probs=t), 1))
    ])


class DMM_TF(keras.Model):
    def __init__(
            self,
            data,
            input_dim=88,
            z_dim=100,
            emission_dim=100,
            transition_dim=200,
            rnn_dim=600,
            rnn_dropout_rate=0.,
            # num_iafs=0,
            # iaf_dim=50,
            t_max=129,
            batch_size=8,
            adam_params=None
    ):
        super().__init__()
        if adam_params is None:
            adam_params = {'learning_rate': .0003, 'beta_1': .96, 'beta_2': .999}
        self.t_max = t_max
        self.batch_size = batch_size
        self.rnn_dim = rnn_dim
        self.z_dim = z_dim
        self.annealing_factor = 0.2  # 1e-3
        self.iteration = 0
        self.data_size = len(data.data)
        self.total_data_points = data.mask.v.sum()
        self.sample_size = 5

        self.emitter = Emitter(input_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, transition_dim)
        self.combiner = Combiner(z_dim, rnn_dim)
        self.rnn = kl.SimpleRNN(
            rnn_dim,
            activation='relu',
            return_sequences=True,
            return_state=False,
            dropout=rnn_dropout_rate,
            input_shape=[input_dim]
        )
        self.z0 = tf.Variable(tf.random.normal([z_dim]), dtype='float32')
        self.zq0 = tf.Variable(tf.random.normal([z_dim]), dtype='float32')
        self.h_0 = tf.Variable(tf.random.normal([rnn_dim]), dtype='float32')

        self.optimizer = keras.optimizers.Adam(**adam_params)

    def guide(self, x, mask):
        batch_size = x.shape[0]
        x_reversed = x[..., ::-1, :]
        h_0 = tf.broadcast_to(self.h_0, [batch_size, self.rnn_dim])
        rnn_output = self.rnn(x_reversed, initial_state=h_0)[..., ::-1, :] * mask[..., None]  # b * t * rnn
        rnn_output = rnn_output  # 1 * b * t * rnn

        zq = tf.broadcast_to(self.zq0, [batch_size, self.t_max, self.z_dim])  # b * t * z

        def distribution_fn(x):  # x: s * b * t * z
            zz = tf.broadcast_to(self.zq0, x[..., :1, :].shape)  # s * b, t * z
            x = tf.concat([zz, x[..., :-1, :]], axis=-2)
            z_loc, z_scale = self.combiner([x, rnn_output])  # s * b * t * z
            return Ind(tfd.Normal(z_loc, z_scale), 1)

        return tfd.Autoregressive(distribution_fn, zq, self.t_max)

    def prior(self, x):
        batch_size = x.shape[0]
        z = tf.zeros([batch_size, self.t_max, self.z_dim])

        def distribution_fn(z):  # ... * b * t * z
            z0 = tf.broadcast_to(self.z0, z[..., :1, :].shape)
            z = tf.concat([z0, z[..., :-1, :]], axis=-2)
            m, s = self.trans(z)
            # m = ms[..., 0]
            # s = ms[..., 1]
            return Ind(tfd.Normal(m, s), 1)

        return tfd.Autoregressive(distribution_fn, z, num_steps=self.t_max, name='z')
