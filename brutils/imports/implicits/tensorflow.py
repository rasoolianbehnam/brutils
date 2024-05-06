import brutils.utility as ut
import tensorflow as tf
import xarray as xr
import arviz as az
import tensorflow_probability as tfp
from tensorflow_probability.python.internal.structural_tuple import structtuple
from functools import wraps

mth = tf.math

keras = tf.keras
tfd = tfp.distributions
tfb = tfp.bijectors
tfl = tfp.layers
tfli = tf.linalg
einsum = tf.einsum
Root = root = tfd.JointDistributionCoroutine.Root
mcmc = tfp.mcmc
nest = tf.nest


def v(self):
    return self.numpy()


tf.Tensor.v = property(v)
tf.Tensor.T = property(tf.transpose)

fun_names = [
    'expand_dims', 'reshape', 'clip_by_value',
    'clip_by_norm', 'broadcast_to', "split",
]

for fun in fun_names:
    setattr(tf.Tensor, fun, getattr(tf, fun))

for fn_name in dir(mth):
    fn = getattr(mth, fn_name)
    if not callable(fn):
        continue
    setattr(tf.Tensor, fn_name, fn)


def get_wavenet_layers(depth, filters):
    return [
        keras.layers.Conv1D(filters=filters,
                            kernel_size=2,
                            padding="causal",
                            activation="relu",
                            dilation_rate=dilation_rate)
        for dilation_rate in (int(2 ** i) for i in range(depth))
    ]


def get_idx(arr: tf.Tensor, idx, n_batches=0):
    idx = tf.cast(idx, tf.int32)
    dims = list(range(len(arr.shape)))
    new_dims = dims[n_batches:] + dims[:n_batches]
    new_tmp = tf.transpose(arr, new_dims)
    return tf.transpose(tf.gather_nd(new_tmp, idx.T))


def outer(a, b):
    return a.reshape([-1, 1]) @ b.reshape([1, -1])


def inner(a, b):
    return a.reshape([1, -1]) @ b.reshape([-1, 1])


def Ind(d, reinterpreted_batch_ndims=1, **kwargs):
    return tfd.Independent(d, reinterpreted_batch_ndims=reinterpreted_batch_ndims, **kwargs)


class TfIdx:
    def __init__(self):
        pass

    def __getitem__(self, args):
        x = args[0]
        args = args[1:]
        l = tf.range(tf.reduce_prod(x.shape)).reshape(x.shape)
        idx = l.__getitem__(args)
        idxUnravelled = tf.unravel_index(idx.reshape(-1), x.shape).T
        return idxUnravelled

    def get(self, x, idx):
        return tf.gather_nd(x, idx)

    def __setitem__(self, args, b):
        x = args[0]
        idx = self.__getitem__(args)
        self.x = tf.tensor_scatter_nd_update(
            x,
            idx,
            b.reshape(-1)
        )


def rt(self):
    return root(self)


def concat_shapes(shapes):
    return tf.concat(list(shapes), axis=0)


def broadcast_to(y, shapes):
    return tf.broadcast_to(y, concat_shapes(shapes))


@ut.RegisterWithClass(tf.Tensor)
@wraps(tf.tensor_scatter_nd_update)
def set(self, *args, **kwargs):
    return tf.tensor_scatter_nd_update(self, *args, **kwargs)


@ut.RegisterWithClass(tfd.Distribution)
def expand(self, size, name=None):
    name = name or self.name
    return tfd.Sample(self, size, name=name)


@ut.RegisterWithClass(tfd.Distribution)
def Sample(self, size, name=None):
    name = name or self.name
    return tfd.Sample(self, size, name=name)


@ut.RegisterWithClass(tfd.Distribution)
def event(self, n_dims=1, name=None):
    name = name or self.name
    return Ind(self, n_dims, name=name)


def prepare_idx(idx):
    if isinstance(idx, tuple):
        idx = [x for x in idx]
    else:
        idx = idx
    return idx


class TensorSetter:
    def __init__(self, tensor, idx):
        self.tensor = tensor
        self.idx = idx

    def get(self):
        return tf.gather_nd(self.tensor, self.idx)

    def set(self, value):
        return tf.tensor_scatter_nd_update(self.tensor, [self.idx], [value])


class TensorGetter:
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, idx):
        return TensorSetter(self.tensor, prepare_idx(idx))


class At:
    def __init__(self, storage_name):
        self.storage_name = storage_name

    def __set__(self, instance, value):
        raise ValueError()

    def __get__(self, instance, owner):
        return TensorGetter(instance)


tf.Tensor.at = At("at")

tf.Tensor.set = tf.tensor_scatter_nd_update
tf.Tensor.get_idx = get_idx
tf.outer = outer
tf.inner = inner
tf.array = tf.constant
tfd.Distribution.root = property(rt)
