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
import ops
import tensorflow as tf


def ResidualBlock(x,
                  outchannel,
                  train,
                  stride=1,
                  shortcut=None,
                  name="ResidualBlock"):
    with tf.variable_scope(name):
        conv1 = ops.conv2d(x, outchannel,
                           k_h=3, k_w=3,
                           s_h=stride, s_w=stride, scope="conv1")
        bn1 = tf.nn.relu(ops.batch_normal(conv1, train=train, scope="bn1"))
        conv2 = ops.conv2d(bn1, outchannel,
                           k_h=3, k_w=3,
                           s_h=1, s_w=1,
                           with_bias=False, scope="conv2")
        left = ops.batch_normal(conv2, train=train, scope="bn2")
        right = x if shortcut is None else shortcut(x)
        return tf.nn.relu(left + right)


class ResNet():
    def __init__(self):
        self.is_training = tf.placeholder(tf.bool, [])
        with tf.variable_scope("input"):
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
        with tf.variable_scope("pre"):
            conv = ops.conv2d(self.x, output_dim=32,
                              k_h=3, k_w=3,
                              s_h=1, s_w=1,
                              with_bias=False)
            bn = tf.nn.relu(ops.batch_normal(conv, self.is_training))

        with tf.variable_scope("layer1"):
            layer1 = self._make_layer(bn, outchannel=32, block_num=3, train=self.is_training)
        with tf.variable_scope("layer2"):
            layer2 = self._make_layer(layer1, outchannel=64, block_num=1, stride=2, train=self.is_training)
        with tf.variable_scope("layer3"):
            layer3 = self._make_layer(layer2, outchannel=32, block_num=2, train=self.is_training)
        with tf.variable_scope("layer4"):
            layer4 = self._make_layer(layer3, outchannel=128, block_num=1, stride=2, train=self.is_training)
        with tf.variable_scope("layer5"):
            layer5 = self._make_layer(layer4, outchannel=128, block_num=2, train=self.is_training)
        pool = tf.nn.avg_pool(layer5, ksize=[1, 8, 8, 1],
                              strides=[1, 8, 8, 1], padding='SAME')
        p2s = pool.get_shape()
        reshape = tf.reshape(pool, [-1, p2s[1] * p2s[2] * p2s[3]])
        self.fc = ops.linear(reshape, 10)

    def __call__(self, *args, **kwargs):
        return self.x, self.fc, self.is_training

    @staticmethod
    def _make_layer(x,
                    train,
                    outchannel,
                    block_num, stride=1):
        def shortcut(input_):
            with tf.variable_scope("shortcut"):
                conv = ops.conv2d(input_, output_dim=outchannel,
                                  k_w=1, k_h=1, s_w=stride, s_h=stride,
                                  with_bias=False)
                return ops.batch_normal(conv, train)

        x = ResidualBlock(x, outchannel, train, stride,
                          shortcut, name="ResidualBlock0")
        for i in range(1, block_num):
            x = ResidualBlock(x, outchannel,
                              train=train,
                              name="ResidualBlock{}".format(i))
        return x

if __name__ == '__main__':
    resnet = ResNet()
    print(resnet())

