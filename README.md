# 以CIFAR10数据为例的分类器
实验课作业，由于是很经典的分类任务，所以整理了一下记录下来，实际上TensorFlow源码中就有很好的CIFAR10示例（包含单机和分布式多种版本），不过既然要交作业，自己的改起来方便一些。在基础CNN分类的版本外，添加了使用ResNet进行分类的实验，收敛速度远快于基础CNN。<br>
## 一、文件介绍
#### 公用脚本
`ops.py`：resnet网络层封装实现，用于被Advanced_CNN.py和ResNet.py调用<br>
`cifar10_input.py`：数据读入相关函数脚本，包含了对训练数据和测试数据的不同预处理路径设置<br>
`eval_CNN.py`：测试用脚本，读取./images目录下的图片文件，并输出对应的预测结果<br>
#### 基础CNN分类器相关脚本
`Advanced_CNN.py`：使用CNN的分类器，脚本本身包含了网络构建和训练相关节点，可以直接运行训练数据<br>
#### ResNet相关脚本
`ResNet.py`：简易版resnet网络构建函数，由于数据为32×32的，太深的网络不适合这么低的分辨率（这是我最初的想法，实际上Kaggle上就是有人用很深的网络得到了很好的结果），应用的网络层数并不多，不过效果还可以，能达到88%左右的准确率<br>
`Res_Train.py`：resnet的训练函数脚本<br>
## 二、预处理流程
[『TensorFlow』读书笔记_进阶卷积神经网络_下](http://www.cnblogs.com/hellcat/p/8018092.html)<br>
![](https://images2017.cnblogs.com/blog/1161096/201712/1161096-20171210153511911-1208313247.png)
## 三、基础CNN分类器
```bash
python Advanced_CNN.py
```
#### 网络设计
[『TensorFlow』读书笔记_进阶卷积神经网络_上](http://www.cnblogs.com/hellcat/p/8017370.html)<br>
输入24*24的图片<br>
卷积->relu激活->最大池化->bn层<br>
卷积->relu激活->bn层->最大池化<br>
全连接：reshape尺寸->384<br>
全连接：192->10<br>
SoftMax<br>
#### 训练结果
大概训练3000个step(batch_size=128)后稳定在75%左右,损失函数变化如下<br>
![](https://github.com/Hellcatzm/ClassifierForCifar10_TensorFlow/blob/master/%E5%B8%B8%E8%A7%84CNN%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E8%A1%B0%E5%87%8F%E5%9B%BE.png)<br>
## 四、 ResNet分类器
[『PyTorch × TensorFlow』第十七弹_ResNet快速实现](https://www.cnblogs.com/hellcat/p/8521191.html)<br>
```bash
python Res_Train.py
```
为了提高网络性能，我们使用滑动平均处理，不过实际来看没什么效果，bn层倒是效果很明显<br>
最终能达到88%的准确率，损失函数变化如下：<br>
![](https://github.com/Hellcatzm/ClassifierForCifar10_TensorFlow/blob/master/ResNet%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E8%A1%B0%E5%87%8F%E5%9B%BE.png)<br>
## 五、预测图片
将预测图片放在images文件夹下，注意需要时jpg文件格式，然后运行：<br>
```bash
python eval_CNN.py
```
默认使用ResNet的网络预测，可以自己在eval_CNN.py中修改，会按照顺序输出下面的预测标签：<br>
['horse', 'bird', 'ship', 'ship', 'dog', 'cat']<br>
上面是ResNet训练之后的预测结果，都是正确的类别，实际图片对应为：
![](https://github.com/Hellcatzm/ClassifierForCifar10_TensorFlow/blob/master/images/timgh.jpeg)
![](https://github.com/Hellcatzm/ClassifierForCifar10_TensorFlow/blob/master/images/timgb.jpeg)
![](https://github.com/Hellcatzm/ClassifierForCifar10_TensorFlow/blob/master/images/timg.jpeg)
![](https://github.com/Hellcatzm/ClassifierForCifar10_TensorFlow/blob/master/images/u%3D1192425208%2C2822262977%26fm%3D200%26gp%3D0.jpg)
![](https://github.com/Hellcatzm/ClassifierForCifar10_TensorFlow/blob/master/images/u%3D382711506%2C2042792358%26fm%3D200%26gp%3D0.jpg)
![](https://github.com/Hellcatzm/ClassifierForCifar10_TensorFlow/blob/master/images/timgc.jpeg)
全部预测正确，但是使用收敛之后的常规CNN网络进行预测，结果如下：<br>
['airplane', 'bird', 'ship', 'ship', 'dog', 'bird']<br>
仅仅有四张图片预测成功(本脚本中的预测仅取top1)。
## 六、更新日志
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
```python
ema = tf.train.ExponentialMovingAverage(0.99, global_step)
with tf.control_dependencies([train_op]):
    variables_averages_op = ema.apply(tf.trainable_variables())
```
