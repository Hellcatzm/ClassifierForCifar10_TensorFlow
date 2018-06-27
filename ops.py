# Author : hellcat
# Time   : 18-6-20

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
 
import numpy as np
np.set_printoptions(threshold=np.inf)
 
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average


def batch_normal(x, train, eps=1e-05, decay=0.9, affine=True, scope='batch_norm'):
    with tf.variable_scope(scope, default_name='BatchNorm2d'):
        params_shape = x.shape[-1]
        moving_mean = tf.get_variable('mean', params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('variance', params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        def mean_var_with_update():
            mean, variance = tf.nn.moments(x, list(range(len(x.shape) - 1)), name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)

        mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x
    # with tf.variable_scope(scope):
    #     beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
    #     gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
    #     axises = np.arange(len(x.shape) - 1)
    #     batch_mean, batch_var = tf.nn.moments(x, list(axises), name='moments')
    #     ema = tf.train.ExponentialMovingAverage(decay=decay)
    #
    #     def mean_var_with_update():
    #         ema_apply_op = ema.apply([batch_mean, batch_var])
    #         with tf.control_dependencies([ema_apply_op]):
    #             return tf.identity(batch_mean), tf.identity(batch_var)
    #
    #     mean, var = tf.cond(train, mean_var_with_update,
    #                         lambda: (ema.average(batch_mean), ema.average(batch_var)))
    #     normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    # return normed

# def batch_normal(x, epsilon=1e-5, momentum=0.9, train=True, scope='batch_norm'):
#     with tf.variable_scope(scope):
#         return tf.contrib.layers.batch_norm(x,
#                                             decay=momentum,
#                                             # updates_collections=None,
#                                             epsilon=epsilon,
#                                             scale=True,
#                                             is_training=train)


'''
Note: when training, the moving_mean and moving_variance need to be updated.
By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
need to be added as a dependency to the `train_op`. For example:

```python
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
```

One can set updates_collections=None to force the updates in place, but that
can have a speed penalty, especially in distributed settings.
'''


# class batch_norm(object):
#     def __init__(self, epsilon=1e-5, decay=0.9, scope="batch_norm"):
#         with tf.variable_scope(scope):
#             self.epsilon = epsilon
#             self.decay = decay
#             # self.scope = scope
#
#     def __call__(self, x, scope, train=True):
#         return tf.contrib.layers.batch_norm(x,
#                                             decay=self.decay,
#                                             updates_collections=None,
#                                             epsilon=self.epsilon,
#                                             scale=True,
#                                             is_training=train,
#                                             scope=scope)


def concat(tensor_a, tensor_b):
    """
    组合Tensor,注意的是这里tensor_a的宽高应该大于等于tensor_b
    :param tensor_a: 前面的tensor
    :param tensor_b: 后面的tensor
    :return:
    """
    if tensor_a.get_shape().as_list()[1] > tensor_b.get_shape().as_list()[1]:
        return tf.concat([tf.slice(tensor_a,
                                   begin=[0, (int(tensor_a.shape[1]) - int(tensor_b.shape[1])) // 2,
                                          (int(tensor_a.shape[1]) - int(tensor_b.shape[1])) // 2, 0],
                                   size=[int(tensor_b.shape[0]), int(tensor_b.shape[1]),
                                         int(tensor_b.shape[2]), int(tensor_a.shape[3])],
                                   name='slice'),
                          tensor_b],
                         axis=3, name='concat')

    elif tensor_a.get_shape().as_list()[1] < tensor_b.get_shape().as_list()[1]:
        return tf.concat([tensor_a,
                          tf.slice(tensor_b,
                                   begin=[0, (int(tensor_b.shape[1]) - int(tensor_a.shape[1])) // 2,
                                          (int(tensor_b.shape[1]) - int(tensor_a.shape[1])) // 2, 0],
                                   size=[int(tensor_a.shape[0]), int(tensor_a.shape[1]),
                                         int(tensor_a.shape[2]), int(tensor_b.shape[3])],
                                   name='slice')],
                         axis=3, name='concat')
    else:
        return tf.concat([tensor_a, tensor_b], axis=3)


def conv_cond_concat(x, y):
    """
    广播并连接向量,用于ac_gan的标签对矩阵拼接
    :param x: features，例如shape：[n,16,16,128]
    :param y: 扩暂维度后的标签，例如shape：[n,1,1,10]
    :return: 拼接后features，例如：[n,16,16,138]
    """
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], axis=3)


def conv2d(input_, output_dim,
           k_h=5, k_w=5, s_h=2, s_w=2, stddev=0.02,
           scope="conv2d", with_w=False, with_bias=True, wl=0.001):
    """
    卷积网络封装
    :param input_:
    :param output_dim: 输出的feature数目
    :param k_h:
    :param k_w:
    :param s_h:
    :param s_w:
    :param stddev:
    :param scope:
    :param with_w:
    :param with_bias: 是否含有bias层
    :param wl:
    :return:
    """

    with tf.variable_scope(scope):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if wl is not None:
            """
            tf.multiply和tf.matmul区别
            解析：
                （1）tf.multiply是点乘，即Returns x * y element-wise.
                （2）tf.matmul是矩阵乘法，即Multiplies matrix a by matrix b,  producing a * b.
            """
            weight_loss = tf.multiply(tf.nn.l2_loss(w), wl, name='weight_loss')
            tf.add_to_collection('losses', weight_loss)
        # tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, s_h, s_w, 1], padding='SAME')
        if with_bias:
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        else:
            biases = None

    if with_w:
        return conv, w, biases
    else:
        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, s_h=2, s_w=2, stddev=0.02,
             scope="deconv2d", with_w=False):
    """
    转置卷积网络封装
    :param input_:
    :param output_shape: 输出的shape
    :param k_h:
    :param k_w:
    :param s_h:
    :param s_w:
    :param stddev:
    :param scope:
    :param with_w:
    :return:
    """
    with tf.variable_scope(scope):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, s_h, s_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, s_h, s_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2):
    """
    Leak_Relu层封装
    :param x:
    :param leak:
    :return:
    """
    return tf.maximum(x, leak * x)


def linear(input_, output_size,
           stddev=0.02, bias_start=0.0,
           scope=None, with_w=False):
    """
    全连接层封装
    :param input_:
    :param output_size: 输出节点数目
    :param scope:
    :param stddev:
    :param bias_start: 使用常数初始化偏执，常数值设定
    :param with_w: 返回是否返回参数Variable
    :return:
    """
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))

        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

