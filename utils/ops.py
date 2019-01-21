import tensorflow as tf
from keras import backend as K
import numpy as np


def diff(a, n=1, axis=-1):
    if axis == -1:
        axis = K.ndim(a) - 1
    a_aug = K.tile(K.expand_dims(a, axis=1), [1, n] + [1] * (K.ndim(a) - 1))
    pattern = (axis,) + tuple(set(range(K.ndim(a))) - {axis})
    inv_pattern = tuple(range(1, axis + 1)) + (0,) + tuple(range(axis + 1, K.ndim(a)))
    def step(inputs, states):
        prev_output = states[0]
        t = states[1]
        t_int = K.cast(t[0], 'int32')
        prev_output_aug = K.permute_dimensions(prev_output, pattern)
        inputs_aug = K.permute_dimensions(inputs, pattern)
        output_aug = K.switch(K.all(t_int > 0),
                              K.concatenate([inputs_aug[:t_int], prev_output_aug[1:] - prev_output_aug[:-1]], axis=0),
                              K.concatenate([inputs_aug[:1], inputs_aug[1:] - inputs_aug[:-1]], axis=0))
        output = K.permute_dimensions(output_aug, inv_pattern)
        return output, [output, t + 1]
    d_aug = K.permute_dimensions(K.rnn(step, a_aug, [K.zeros_like(a), K.zeros((1,))])[0], pattern)
    d = K.permute_dimensions(d_aug[-(K.shape(a)[axis] - n):], inv_pattern)
    return d


def cumsum(a, n=1, axis=-1):
    if axis == -1:
        axis = K.ndim(a) - 1
    a_aug = K.tile(K.expand_dims(a, axis=1), [1, n] + [1] * (K.ndim(a) - 1))
    pattern = (axis,) + tuple(set(range(K.ndim(a))) - {axis})
    inv_pattern = tuple(range(1, axis + 1)) + (0,) + tuple(range(axis + 1, K.ndim(a)))
    def step(inputs, states):
        prev_output = states[0]
        t = states[1]
        t_int = K.cast(t[0], 'int32')
        prev_output_aug = K.permute_dimensions(prev_output, pattern)
        inputs_aug = K.permute_dimensions(inputs, pattern)
        output_aug = K.switch(K.all(t_int > 0), K.cumsum(prev_output_aug, axis=0), K.cumsum(inputs_aug, axis=0))
        output = K.permute_dimensions(output_aug, inv_pattern)
        return output, [output, t + 1]
    c_aug = K.permute_dimensions(K.rnn(step, a_aug, [K.zeros_like(a), K.zeros((1,))])[0], pattern)
    c = K.permute_dimensions(c_aug[-(K.shape(a)[axis] - n):], inv_pattern)
    return c


def n_p(Z, p, axis, epsilon=None):
    if epsilon is None:
        epsilon = K.epsilon()
    return K.pow(K.sum(K.pow(K.abs(Z), p), axis=axis) + epsilon, 1. / p)

def floor_div(x, y):
    return K.shape(K.ones((x,))[::y])[0]

def group_norms(inputs, groups, axis, norm=2, epsilon=None):
    if axis == -1:
        axis = K.ndim(inputs) - 1
    if epsilon is None:
        epsilon = K.epsilon()
    inputs_group = tf.split(inputs, num_or_size_splits=groups, axis=axis)
    inputs_group = K.concatenate([K.expand_dims(t, axis=axis) for t in inputs_group], axis=axis)
    return n_p(inputs_group, p=norm, axis=(axis + 1), epsilon=epsilon)
    
def ksparse(x, k, axis, alpha=1, absolute=False):
    if isinstance(axis, int):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    axis_complement = tuple(set(range(K.ndim(x))) - set(axis))
    shape_reduce = K.prod([K.shape(x)[j] for j in axis])
    _k = K.minimum(K.in_train_phase(k, alpha * k), shape_reduce)
    inputs_permute_dimensions = K.permute_dimensions(x, axis_complement + axis)
    inputs_permute_dimensions_reshape = K.reshape(inputs_permute_dimensions, (-1, shape_reduce))
    if absolute is True:
        inputs_permute_dimensions_reshape = K.abs(inputs_permute_dimensions_reshape)
    _, indices = tf.nn.top_k(inputs_permute_dimensions_reshape, _k)
    scatter_indices = K.concatenate([(K.arange(K.shape(inputs_permute_dimensions_reshape)[0])[:, None] * K.ones((1, _k), dtype='int32'))[:, :, None], indices[:, :, None]])
    scatter_updates = K.ones((K.shape(inputs_permute_dimensions_reshape)[0], _k))
    mask_permute_dimensions_reshape = K.cast(tf.scatter_nd(scatter_indices, scatter_updates, K.shape(inputs_permute_dimensions_reshape)), K.floatx())
    mask_permute_dimensions = K.reshape(mask_permute_dimensions_reshape, K.shape(inputs_permute_dimensions))
    mask = K.permute_dimensions(mask_permute_dimensions, tuple(np.argsort(axis_complement + axis)))
    return mask * x

def group_ksparse(x, groups, k, axis_group, axis_sparse, norm=2, alpha=1, epsilon=None):
    if isinstance(axis_group, int):
        axis_group = (axis_group,)
    elif isinstance(axis_group, list):
        axis_group = tuple(axis_group)
    if isinstance(axis_sparse, int):
        axis_sparse = (axis_sparse,)
    elif isinstance(axis_sparse, list):
        axis_sparse = tuple(axis_sparse)
    assert(1 - bool(set(axis_group) & set(axis_sparse)))
    if epsilon is None:
        epsilon = K.epsilon()
    axis_complement = tuple(set(range(K.ndim(x))) - set(axis_group) - set(axis_sparse))
    shape_reduce_group = K.prod([K.shape(x)[j] for j in axis_group])
    shape_reduce_sparse = K.prod([K.shape(x)[j] for j in axis_sparse])
    _k = K.minimum(K.in_train_phase(k, alpha * k), shape_reduce_sparse)
    inputs_permute_dimensions = K.permute_dimensions(x, axis_complement + axis_sparse + axis_group)
    inputs_permute_dimensions_reshape = K.reshape(inputs_permute_dimensions, (-1, shape_reduce_sparse, shape_reduce_group))
    norm_group_permute_dimensions_reshape = group_norms(inputs=inputs_permute_dimensions_reshape, groups=groups, axis=-1, norm=norm, epsilon=epsilon)
    norm_group_permute_dimensions_reshape = K.permute_dimensions(norm_group_permute_dimensions_reshape, (0, 2, 1))
    norm_group_permute_dimensions_reshape = K.reshape(norm_group_permute_dimensions_reshape, (-1, shape_reduce_sparse))
    _, indices = tf.nn.top_k(norm_group_permute_dimensions_reshape, _k)
    scatter_indices = K.concatenate([(K.arange(K.shape(norm_group_permute_dimensions_reshape)[0])[:, None] * K.ones((1, _k), dtype='int32'))[:, :, None], indices[:, :, None]])
    scatter_updates = K.ones((K.shape(norm_group_permute_dimensions_reshape)[0], _k))
    mask_group_permute_dimensions_reshape = K.cast(tf.scatter_nd(scatter_indices, scatter_updates, K.shape(norm_group_permute_dimensions_reshape)), K.floatx())
    mask_group_permute_dimensions_reshape = K.reshape(mask_group_permute_dimensions_reshape, (-1, groups, shape_reduce_sparse))
    mask_group_permute_dimensions_reshape = K.permute_dimensions(mask_group_permute_dimensions_reshape, (0, 2, 1))
    mask_permute_dimensions_reshape = (mask_group_permute_dimensions_reshape[:, :, :, None] * K.ones((1, 1, 1, floor_div(shape_reduce_group, groups))))
    mask_permute_dimensions = K.reshape(mask_permute_dimensions_reshape, K.shape(inputs_permute_dimensions))
    mask = K.permute_dimensions(mask_permute_dimensions, tuple(np.argsort(axis_complement + axis_sparse + axis_group)))
    return mask * x

def kron(mat1, mat2):
    #Computes the Kronecker product of two matrices.
    mat1_rsh = K.reshape(mat1, [K.shape(mat1)[0], 1, K.shape(mat1)[1], 1])
    mat2_rsh = K.reshape(mat2, [1, K.shape(mat2)[0], 1, K.shape(mat2)[2]])
    return K.reshape(mat1_rsh * mat2_rsh, [K.shape(mat1)[0] * K.shape(mat2)[0], K.shape(mat1)[1] * K.shape(mat2)[1]])

