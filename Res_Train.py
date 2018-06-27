# Author : hellcat
# Time   : 18-6-20

import os
import math
import time
import numpy as np
import tensorflow as tf

import ops
from ResNet import ResNet
# from ResNet_v2 import ResNet
import cifar10_input as cifar10_input

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

data_dir = './cifar-10/'

max_steps = 20000
batch_size = 128

IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


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
    # ———————————————————————————读取图片并预处理————————————————————————————————————
    images_train,  labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                                 batch_size=batch_size,
                                                                 image_size=32)
    images_test,  labels_test = cifar10_input.inputs(eval_data=True,
                                                     data_dir=data_dir,
                                                     batch_size=batch_size,
                                                     image_size=32)

    # ——获取网络节点———
    resnet = ResNet()
    image_holder, logits, is_training = resnet()
    label_holder = tf.placeholder(tf.int32,  [None])
    # 输出结果top_k准确率，默认为1
    top_k_op = tf.nn.in_top_k(logits,  label_holder,  1)
    loss = loss_fn(logits,  label_holder)

    # —————————————————————————learning rate衰减 & 优化器节点——————————————————————————————————————————————————
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    # num_batches_per_epoch = (cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
    #                          batch_size)
    # decay_steps = int(num_batches_per_epoch * 10)
    lr = tf.train.exponential_decay(1e-3,
                                    global_step,
                                    3000,  # decay_steps,
                                    0.1,
                                    staircase=True)
    tf.summary.scalar("learning_rate", lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # minimize需要接收global_step才会更新lr
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

    # ——————————————————添加滑动平均————————————————————————————
    ema = tf.train.ExponentialMovingAverage(0.99, global_step)
    with tf.control_dependencies([train_op]):
        variables_averages_op = ema.apply(tf.trainable_variables())

    def test_cifar10(session, training=False):
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
                                                           is_training: training})
            true_count += np.sum(predictions)
            steps += 1
        accuracy = true_count / total_sample_count
        print("[*] test accuracy is {:.3f}".format(accuracy))
        return accuracy

    # ——————————训练准备—————————————
    summary = tf.summary.merge_all()
    sess = tf.InteractiveSession(config=config)
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    saver = tf.train.Saver()
    model_path = "./logs_res"
    writer = tf.summary.FileWriter(model_path, sess.graph)
    if not os.path.exists(os.path.join(model_path, "model")):
        os.makedirs(os.path.join(model_path, "model"))
    ckpt = tf.train.get_checkpoint_state(os.path.join(model_path, "model"))
    if ckpt is not None:
        print("[*] Success to read {}".format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("[*] Failed to find a checkpoint")

    # ——训练——
    acc = []
    for step in range(max_steps):
        start_time = time.time()
        image_batch,  label_batch = sess.run([images_train,  labels_train])
        _,  loss_value, summary_str = sess.run([variables_averages_op,  loss, summary],
                                               feed_dict={image_holder: image_batch,
                                                          label_holder: label_batch,
                                                          is_training: True})
        writer.add_summary(summary_str, step)
        duration = time.time() - start_time
        if step % 10 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            format_str = "[*] step %d,  loss=%.2f (%.1f examples/sec; %.3f sec/batch)"
            print(format_str % (step,  loss_value,  examples_per_sec,  sec_per_batch))
        if step % 100 == 0:
            accuracy_step = test_cifar10(sess, training=False)
            acc.append('{:.3f}'.format(accuracy_step))
            print(acc)
        if step % 500 == 0 and step != 0:
            saver.save(sess, os.path.join(model_path, "model/DCGAN.model"), global_step=step)

    # ———————————使用滑动平均参数测试准确率———————————————
    saver = tf.train.Saver(ema.variables_to_restore())
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt is not None:
        print("[*] Success to read {}".format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    test_cifar10(sess, training=False)

    # —————清理现场——————
    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    main()

