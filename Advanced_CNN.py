# Author : Hellcat
# Time   : 2017/12/8
import os
import math
import time
import numpy as np
import tensorflow as tf

import ops
import cifar10_input as cifar10_input

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

data_dir = './cifar-10/'

max_steps = 5000
batch_size = 128

IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def variable_with_weight_loss(shape,  stddev,  wl):
    """
    权参数初始化，会使用L2正则化
    :param shape: 权重尺寸
    :param stddev: 标准差
    :param wl: L2项稀疏
    :return: 权重变量
    """
    var = tf.Variable(tf.truncated_normal(shape,  stddev=stddev))
    if wl is not None:
        """
        tf.multiply和tf.matmul区别
        解析：
            （1）tf.multiply是点乘，即Returns x * y element-wise.
            （2）tf.matmul是矩阵乘法，即Multiplies matrix a by matrix b,  producing a * b.
        """
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses',  weight_loss)
    return var


def inference(image_holder, is_training):
    # 卷积->relu激活->最大池化->标准化
    weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.)
    bias1 = tf.Variable(tf.constant(0., shape=[64]))
    kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
    pool1 = tf.nn.max_pool(conv1,  ksize=[1, 3, 3, 1],  strides=[1, 2, 2, 1], padding='SAME')
    norm1 = ops.batch_normal(pool1, train=is_training, scope='BN_1')

    # 卷积->relu激活->标准化->最大池化
    weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.)
    bias2 = tf.Variable(tf.constant(0., shape=[64]))
    kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
    norm2 = ops.batch_normal(conv2, train=is_training, scope='BN_2')
    pool2 = tf.nn.max_pool(norm2,  ksize=[1, 3, 3, 1],  strides=[1, 2, 2, 1],  padding='SAME')

    # 全连接：reshape尺寸->384
    p2s = pool2.get_shape()
    reshape = tf.reshape(pool2, [-1, p2s[1]*p2s[2]*p2s[3]])
    dim = reshape.get_shape()[1].value
    weight3 = variable_with_weight_loss(shape=[dim,  384],  stddev=0.04,  wl=0.004)
    bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
    local3 = tf.nn.relu(tf.matmul(reshape, weight3)+bias3)

    # 全连接：384->192
    weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
    bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
    """
    tf.nn.bias_add 是 tf.add 的一个特例
    二者均支持 broadcasting（广播机制），也即两个操作数最后一个维度保持一致。
    除了支持最后一个维度保持一致的两个操作数相加外，tf.add 还支持第二个操作数是一维的情况
    """
    local4 = tf.nn.relu(tf.nn.bias_add(tf.matmul(local3, weight4),  bias4))

    # 全连接：192->10
    weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192., wl=0.)
    bias5 = tf.Variable(tf.constant(0., shape=[10]))
    logits = tf.add(tf.matmul(local4, weight5), bias5, name="logits")
    # print(logits)
    return logits


def loss_fn(logits,  labels):
    """
    loss函数计算
    :param logits: 网络输出结果
    :param labels: 真实标签
    :return: 
    """
    labels = tf.cast(labels, tf.int64)
    # 使用SoftMax交叉熵函数，loss计算自带softmax层
    # 对比下面的print可以得知输出的是128张图片各自的交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, 
                                                                   name='cross_entropy_per_example')
    # print('交叉熵：', cross_entropy.get_shape()) # (128, )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    tf.add_to_collection('losses',  cross_entropy_mean)
    # tf.add_n():多项连加
    tutal_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    tf.summary.scalar("loss", tutal_loss)
    return tutal_loss


def main():
    # 读取图片并预处理
    images_train,  labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                                 batch_size=batch_size,
                                                                 image_size=24)
    images_test,  labels_test = cifar10_input.inputs(eval_data=True,
                                                     data_dir=data_dir,
                                                     batch_size=batch_size,
                                                     image_size=24)
    # print(images_test, labels_test)
    # 输入：24*24的RGB三色通道图片
    image_holder = tf.placeholder(tf.float32,  [None, 24, 24, 3])
    # print(image_holder)
    label_holder = tf.placeholder(tf.int32,  [None])
    training_holder = tf.placeholder(tf.bool, [])
    logits = inference(image_holder, training_holder)
    loss = loss_fn(logits,  label_holder)

    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    # num_batches_per_epoch = (cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
    #                          batch_size)
    # decay_steps = int(num_batches_per_epoch * 350.0)
    lr = tf.train.exponential_decay(1e-3,
                                    global_step,
                                    300,
                                    0.96,
                                    staircase=True)
    tf.summary.scalar("learning rate", lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # minimize需要接收global_step才会更新lr
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

    # 输出结果top_k准确率，默认为1
    top_k_op = tf.nn.in_top_k(logits,  label_holder,  1)

    def test_cifar10(session):
        # 测试部分
        num_examples = 10000

        num_iter = int(math.ceil(num_examples / batch_size))
        true_count = 0
        total_sample_count = num_iter * batch_size
        steps = 0
        while steps < num_iter:
            images, labels = session.run([images_test, labels_test])
            predictions = session.run(top_k_op, feed_dict={image_holder: images,
                                                           label_holder: labels,
                                                           training_holder: True})
            true_count += np.sum(predictions)
            steps += 1
        accuracy = true_count / total_sample_count
        print("[*] test accuracy is {:.3f}".format(accuracy))

    # 训练部分
    summary = tf.summary.merge_all()
    sess = tf.InteractiveSession(config=config)
    tf.global_variables_initializer().run()
    # 启动数据增强队列
    tf.train.start_queue_runners()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("./logs", sess.graph)
    if not os.path.exists("./logs/model"):
        os.makedirs("./logs/model")
    ckpt = tf.train.get_checkpoint_state("./logs/model")
    if ckpt is not None:
        print("[*] Success to read {}".format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("[*] Failed to find a checkpoint")

    for step in range(max_steps):
        start_time = time.time()
        image_batch,  label_batch = sess.run([images_train,  labels_train])
        _,  loss_value, summary_str = sess.run([train_op,  loss, summary],
                                               feed_dict={image_holder: image_batch,
                                                          label_holder: label_batch,
                                                          training_holder: False})
        writer.add_summary(summary_str, step)
        duration = time.time() - start_time
        if step % 10 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            format_str = "[*] step %d,  loss=%.2f (%.1f examples/sec; %.3f sec/batch)"
            print(format_str % (step,  loss_value,  examples_per_sec,  sec_per_batch))
        if step % 100 == 0:
            test_cifar10(sess)
        if step % 500 == 0 and step != 0:
            saver.save(sess, "./logs/model/DCGAN.model", global_step=step)
    sess.close()


if __name__ == '__main__':
    main()
