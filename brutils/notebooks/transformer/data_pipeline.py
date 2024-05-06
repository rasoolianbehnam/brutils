import numpy as np
import tensorflow as tf
import brutils.utility as ut
# from tensorflow.python.data.ops.readers import TFRecordDatasetV2
# from tensorflow.python.data.ops.map_op import _MapDataset
# from tensorflow.python.data.ops.filter_op import _FilterDataset


def create_decode_fn(string_lookup, max):
    def decode(x):
        res = tf.io.parse_single_example(
            x,
            features={
                'pool_ids': tf.io.VarLenFeature(tf.string),
                'reach_i': tf.io.VarLenFeature(tf.int64),
                'reach': tf.io.FixedLenFeature([], tf.int64),

            })
        return (
            (
                string_lookup(tf.sparse.to_dense(res['pool_ids'])),
                tf.cast(tf.sparse.to_dense(res['reach_i'])/max, tf.float32),
            ),
            tf.cast(res['reach']/max, tf.float32)
        )
    return decode


def prepare(ds):
    return (
        ds
            .bucket_by_sequence_length(
            lambda sr_i, r: tf.shape(sr_i[0])[0],
            bucket_boundaries=[100, 500],
            bucket_batch_sizes=[256, 128, 32]
        )
    )