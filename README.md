# 以CIFAR10数据为例的分类器
实验课作业，由于是很经典的分类任务，所以整理了一下记录下来，实际上TensorFlow源码中就有很好的CIFAR10示例（包含单机和分布式多种版本），不过既然要交作业，自己的改起来方便一些。在基础CNN分类的版本外，添加了使用ResNet进行分类的实验，收敛速度远快于基础CNN。<br>
## 更新日志
#### 18.6.20
##### 1、修改了张量流shape限制方式以适应不同batch的输入
为了能够满足测试时使用不同batch_size的数据，将输入做如下替换：<br>
```python
image_holder = tf.placeholder(tf.float32, [batch_size,24,24,3])  # 原

image_holder = tf.placeholder(tf.float32,  [None, 24, 24, 3])    # 新
```
为了适应上面改动，全连接层前的处理修改如下，<br>
```python
reshape = tf.reshape(pool2,[batch_size,-1])                      # 原

p2s = pool2.get_shape()                                          # 新
reshape = tf.reshape(pool2, [-1, p2s[1]*p2s[2]*p2s[3]])          # 新
```
##### 2、添加了梯度衰减
```python
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
    # minimize需要接收global_step才会更新lr的衰减
    train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
```
##### 3、将lrn层替换为bn层
提高网络效果，效果很好
##### 4、添加ResNet脚本
准确率由75%左右提高到88%
##### 5、添加滑动平均
