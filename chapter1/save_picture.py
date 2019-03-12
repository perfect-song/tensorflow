from tensorflow.examples.tutorials.mnist import input_data

import scipy.misc

import os

import numpy as np

#读取MNIST数据集，如果不存在会先下载
mnist = input_data.read_data_sets('G:/PyWork/data/mnist/', one_hot=True)

#吧原始图像保存在mnist/raw/文件夹下

save_dir = 'G:/PyWork/data/mnist/raw/'

if os.path.exists(save_dir) is False:
    print('创建目录')
    os.makedirs(save_dir)

# #保存前20张图片
# for i in range(20):
#     # mnist.train.images[i,:]表示第i张图片 ,后面表示图像
#     image_array = mnist.train.images[i,:]
#      #tensorflow 中的mnist照片是以784维向量表示的，重新还原回28*28维图像
#
#     image_array = image_array.reshape(28,28)
#
#     #保存文件格式
#     filename = save_dir+'minst_train_%{}.jpg'.format(i)
#
#     # 先用scipy,misc.toimage转换成图像，在调用svae保存
#     scipy.misc.toimage(image_array,cmin=0.0,cmax=1.0).save(filename)



# print(list(mnist.train.labels[1,:]).index(1)) # 3 说明第2幅图片是数字3

##输出前20张图片的标签
for i in range(20):
    one_hot_label = mnist.train.labels[i,:]
    label = np.argmax(one_hot_label)

    print('mnist train %{}.jpg lable is {}'.format(i,label))

