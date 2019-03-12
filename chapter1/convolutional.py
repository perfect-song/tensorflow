# coding: utf-8
'''

卷积神经网络对mnist手写数据集进行分类


tf.truncated_normal(shape,mean,stddev) 均值为mean 标准差为stddev 的正态分布。这是一个截断产生正态分布的函数，就是说如果产生的正态分布的值如果与均值的差值大于 两倍标准差，就重新生成

tf.constant(value,shape) 生成给定value值的常量
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('G:/pywork/data/mnist/', one_hot=True)  ##加载数据

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])  ##将784维向量表示成28*28的图像


##定义网络层参数
def var_w(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.01)  ## W 初始化为方程为0.01的正态分布
    return tf.Variable(initial)


def var_b(shape):
    # initial = tf.zeros(shape)
    initial = tf.constant(0.1, shape=shape)  ##生成给定0.1的常量
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 第一层
w_conv1 = var_w([5, 5, 1, 32])
b_conv1 = var_b([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层
w_conv2 = var_w([5, 5, 32, 64])
b_conv2 = var_b([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

##全连接成
w_fc1 = var_w([7 * 7 * 64, 1024])
b_fc1 = var_b([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  ##拉平，变成一维
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

##使用dropout keep_prob是占位符
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = var_w([1024, 10])
b_fc2 = var_b([10])
y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = mnist.train.next_batch(50)

    if i % 100==0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0],y_:batch[1],keep_prob: 1.0})
        print('step {}， train accuracy is {}'.format(i,train_accuracy))

    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

print("test accuracy {}".format(accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})))


