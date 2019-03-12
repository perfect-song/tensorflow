import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os

'''
tf.argmax(input, axis,name,dimension)取出数组中最大值下标，1表示维度,行最大还列最大
tf.cast(x,dtype,name) 将x转换成dtype类型
tf.equal(x,y)表示 x，y是否相等，返回True or False [true,false,true,false]

'''
mnist = input_data.read_data_sets('G:/pywork/data/mnist/', one_hot=True)

##创建占位符
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

##定于变量

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))  ##定义交叉熵损失函数

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  ##梯度下降优化交叉熵损失函数，学习率设置为0.01

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()  ##变量初始化

for _ in range(1000):  ##迭代1000次更新参数w和b
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

##正确预测结结果
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
