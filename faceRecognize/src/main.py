# -*- coding:UTF-8 -*-
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K

#------------用于绘制模型细节，可选--------------#
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#------------------------------------------------#

K.set_image_data_format('channels_first')

import time
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import fr_utils
from inception_blocks import *
import time
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import nst_utils



def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    根据公式（4）实现三元组损失函数

    参数：
        y_true -- true标签，当你在Keras里定义了一个损失函数的时候需要它，但是这里不需要。
        y_pred -- 列表类型，包含了如下参数：
            anchor -- 给定的“anchor”图像的编码，维度为(None,128)
            positive -- “positive”图像的编码，维度为(None,128)
            negative -- “negative”图像的编码，维度为(None,128)
        alpha -- 超参数，阈值

    返回：
        loss -- 实数，损失的值
    """
    #获取anchor, positive, negative的图像编码
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    #第一步：计算"anchor" 与 "positive"之间编码的距离，这里需要使用axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis=-1)

    #第二步：计算"anchor" 与 "negative"之间编码的距离，这里需要使用axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),axis=-1)

    #第三步：减去之前的两个距离，然后加上alpha
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)

    #通过取带零的最大值和对训练样本的求和来计算整个公式
    loss = tf.reduce_sum(tf.maximum(basic_loss,0))

    return loss
def verify(image_path, identity, database, model):
    """
    对“identity”与“image_path”的编码进行验证。

    参数：
        image_path -- 摄像头的图片。
        identity -- 字符类型，想要验证的人的名字。
        database -- 字典类型，包含了成员的名字信息与对应的编码。
        model -- 在Keras的模型的实例。

    返回：
        dist -- 摄像头的图片与数据库中的图片的编码的差距。
        is_open_door -- boolean,是否该开门。
    """
    #第一步：计算图像的编码，使用fr_utils.img_to_encoding()来计算。
    encoding = fr_utils.img_to_encoding(image_path, model)

    #第二步：计算与数据库中保存的编码的差距
    dist = np.linalg.norm(encoding - database[identity])

    #第三步：判断是否打开门
    if dist < 0.7:
        print("欢迎 " + str(identity) + "回家！")
        is_door_open = True
    else:
        print("经验证，您与" + str(identity) + "不符！")
        is_door_open = False

    return dist, is_door_open
def who_is_it(image_path, database,model):
    """
    根据指定的图片来进行人脸识别

    参数：
        images_path -- 图像地址
        database -- 包含了名字与编码的字典
        model -- 在Keras中的模型的实例。

    返回：
        min_dist -- 在数据库中与指定图像最相近的编码。
        identity -- 字符串类型，与min_dist编码相对应的名字。
    """
    #步骤1：计算指定图像的编码，使用fr_utils.img_to_encoding()来计算。
    encoding = fr_utils.img_to_encoding(image_path, model)

    #步骤2 ：找到最相近的编码
    ## 初始化min_dist变量为足够大的数字，这里设置为100
    min_dist = 100

    ## 遍历数据库找到最相近的编码
    for (name,db_enc) in database.items():
        ### 计算目标编码与当前数据库编码之间的L2差距。
        dist = np.linalg.norm(encoding - db_enc)

        ### 如果差距小于min_dist，那么就更新名字与编码到identity与min_dist中。
        if dist < min_dist:
            min_dist = dist
            identity = name

    # 判断是否在数据库中
    if min_dist > 0.7:
        print("抱歉，您的信息不在数据库中。")

    else:
        print("姓名" + str(identity) + "  差距：" + str(min_dist))

    return min_dist, identity
def compute_content_cost(a_C, a_G):
    """
    计算内容代价的函数

    参数：
        a_C -- tensor类型，维度为(1, n_H, n_W, n_C)，表示隐藏层中图像C的内容的激活值。
        a_G -- tensor类型，维度为(1, n_H, n_W, n_C)，表示隐藏层中图像G的内容的激活值。

    返回：
        J_content -- 实数，用上面的公式1计算的值。

    """

    #获取a_G的维度信息
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    #对a_C与a_G从3维降到2维
    a_C_unrolled = tf.transpose(tf.reshape(a_C, [n_H * n_W, n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    #计算内容代价
    #J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    J_content = 1/(4*n_H*n_W*n_C)*tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    return J_content

def gram_matrix(A):
    """
    计算矩阵A的风格矩阵

    参数：
        A -- 矩阵，维度为(n_C, n_H * n_W)

    返回：
        GA -- A的风格矩阵，维度为(n_C, n_C)

    """
    GA = tf.matmul(A, A, transpose_b = True)

    return GA
def compute_layer_style_cost(a_S, a_G):
    """
    计算单隐藏层的风格损失

    参数：
        a_S -- tensor类型，维度为(1, n_H, n_W, n_C)，表示隐藏层中图像S的风格的激活值。
        a_G -- tensor类型，维度为(1, n_H, n_W, n_C)，表示隐藏层中图像G的风格的激活值。

    返回：
        J_content -- 实数，用上面的公式2计算的值。

    """
    #第1步：从a_G中获取维度信息
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    #第2步，将a_S与a_G的维度重构为(n_C, n_H * n_W)
    a_S = tf.transpose(tf.reshape(a_S, [n_H * n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    #第3步，计算S与G的风格矩阵
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    #第4步：计算风格损失
    #J_style_layer = (1/(4 * np.square(n_C) * np.square(n_H * n_W))) * (tf.reduce_sum(tf.square(tf.subtract(GS, GG))))
    J_style_layer = 1/(4*n_C*n_C*n_H*n_H*n_W*n_W)*tf.reduce_sum(tf.square(tf.subtract(GS, GG)))

    return J_style_layer
def compute_style_cost(model, STYLE_LAYERS):
    """
    计算几个选定层的总体风格成本

    参数：
        model -- 加载了的tensorflow模型
        STYLE_LAYERS -- 字典，包含了：
                        - 我们希望从中提取风格的层的名称
                        - 每一层的系数（coeff）
    返回：
        J_style - tensor类型，实数，由公式(2)定义的成本计算方式来计算的值。

    """
    # 初始化所有的成本值
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        #选择当前选定层的输出
        out = model[layer_name]

        #运行会话，将a_S设置为我们选择的隐藏层的激活值
        a_S = sess.run(out)

        # 将a_G设置为来自同一图层的隐藏层激活,这里a_G引用model[layer_name]，并且还没有计算，
        # 在后面的代码中，我们将图像G指定为模型输入，这样当我们运行会话时，
        # 这将是以图像G作为输入，从隐藏层中获取的激活值。
        a_G = out 

        #计算当前层的风格成本
        J_style_layer = compute_layer_style_cost(a_S,a_G)

        # 计算总风格成本，同时考虑到系数。
        J_style += coeff * J_style_layer

    return J_style
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    计算总成本

    参数：
        J_content -- 内容成本函数的输出
        J_style -- 风格成本函数的输出
        alpha -- 超参数，内容成本的权值
        beta -- 超参数，风格成本的权值

    """

    J = alpha * J_content + beta * J_style

    return J
# 第7步：初始化TensorFlow图，进行多次迭代，每次迭代更新生成的图像。
def model_nn(sess, input_image, num_iterations = 200, is_print_info = True, 
             is_plot = True, is_save_process_image = True, 
             save_last_image_to = "output/generated_image.jpg"):
    #初始化全局变量
    sess.run(tf.global_variables_initializer())

    #运行带噪声的输入图像
    sess.run(model["input"].assign(input_image))

    for i in range(num_iterations):
        #运行最小化的目标：
        sess.run(train_step)

        #产生把数据输入模型后生成的图像
        generated_image = sess.run(model["input"])

        if is_print_info and i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("第 " + str(i) + "轮训练," + 
                  "  总成本为:"+ str(Jt) + 
                  "  内容成本为：" + str(Jc) + 
                  "  风格成本为：" + str(Js))
        if is_save_process_image: 
            nst_utils.save_image("output/" + str(i) + ".png", generated_image)

    nst_utils.save_image(save_last_image_to, generated_image)

    return generated_image


if __name__ == '__main__':
    """
         人脸识别人脸验证
    with tf.Session() as test:
        tf.set_random_seed(1)
        y_true = (None, None, None)
        y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
                  tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
                  tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
        loss = triplet_loss(y_true, y_pred)
    
        print("loss = " + str(loss.eval()))
        
    FRmodel = faceRecoModel(input_shape=(3,96,96))
    #开始时间
    start_time = time.clock()
    
    #编译模型
    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    
    #加载权值
    fr_utils.load_weights_from_FaceNet(FRmodel)
    
    #结束时间
    end_time = time.clock()
    
    #计算时差
    minium = end_time - start_time
    
    print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium%60)) + "秒")
    database = {}
    database["danielle"] = fr_utils.img_to_encoding("E:\wuenda\images/danielle.png", FRmodel)
    database["younes"] = fr_utils.img_to_encoding("E:\wuenda\images/younes.jpg", FRmodel)
    database["tian"] = fr_utils.img_to_encoding("E:\wuenda\images/tian.jpg", FRmodel)
    database["andrew"] = fr_utils.img_to_encoding("E:\wuenda\images/andrew.jpg", FRmodel)
    database["kian"] = fr_utils.img_to_encoding("E:\wuenda\images/kian.jpg", FRmodel)
    database["dan"] = fr_utils.img_to_encoding("E:\wuenda\images/dan.jpg", FRmodel)
    database["sebastiano"] = fr_utils.img_to_encoding("E:\wuenda\images/sebastiano.jpg", FRmodel)
    database["bertrand"] = fr_utils.img_to_encoding("E:\wuenda\images/bertrand.jpg", FRmodel)
    database["kevin"] = fr_utils.img_to_encoding("E:\wuenda\images/kevin.jpg", FRmodel)
    database["felix"] = fr_utils.img_to_encoding("E:\wuenda\images/felix.jpg", FRmodel)
    database["benoit"] = fr_utils.img_to_encoding("E:\wuenda\images/benoit.jpg", FRmodel)
    database["arnaud"] = fr_utils.img_to_encoding("E:\wuenda\images/arnaud.jpg", FRmodel)
    verify("E:\wuenda\images/camera_0.jpg","younes",database,FRmodel)
    verify("E:\wuenda\images/camera_2.jpg", "kian", database, FRmodel)
    who_is_it("E:\wuenda\images/camera_0.jpg", database, FRmodel)
    """
    """
    风格
    model = nst_utils.load_vgg_model("E:\wuenda\pretrained-model/imagenet-vgg-verydeep-19.mat")
    print(model)
    content_image = scipy.misc.imread("E:\wuenda\images/louvre.jpg")
    imshow(content_image)
    plt.show()
    tf.reset_default_graph()

    with tf.Session() as test:
        tf.set_random_seed(1)
        a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        J_content = compute_content_cost(a_C, a_G)
        print("J_content = " + str(J_content.eval()))
    
        test.close()
        
    style_image = scipy.misc.imread("E:\wuenda\images/monet_800600.jpg")

    imshow(style_image)
    plt.show()
    tf.reset_default_graph()

    with tf.Session() as test:
        tf.set_random_seed(1)
        A = tf.random_normal([3, 2*1], mean=1, stddev=4)
        GA = gram_matrix(A)
    
        print("GA = " + str(GA.eval()))
    
        test.close()
        
    tf.reset_default_graph()

    with tf.Session() as test:
        tf.set_random_seed(1)
        a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        J_style_layer = compute_layer_style_cost(a_S, a_G)
    
        print("J_style_layer = " + str(J_style_layer.eval()))
    
        test.close()
    STYLE_LAYERS = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)]
    tf.reset_default_graph()

    with tf.Session() as test:
        np.random.seed(3)
        J_content = np.random.randn()    
        J_style = np.random.randn()
        J = total_cost(J_content, J_style)
        print("J = " + str(J))
    
        test.close()
        """
    STYLE_LAYERS = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)]
    #重设图
    tf.reset_default_graph()
    
    #第1步：创建交互会话
    sess = tf.InteractiveSession()
    
    #第2步：加载内容图像(卢浮宫博物馆图片),并归一化图像
    content_image = scipy.misc.imread("E:\wuenda\images/louvre_small.jpg")
    content_image = nst_utils.reshape_and_normalize_image(content_image)
    
    #第3步：加载风格图像(印象派的风格),并归一化图像
    style_image = scipy.misc.imread("E:\wuenda\images/monet.jpg")
    style_image = nst_utils.reshape_and_normalize_image(style_image)
    
    #第4步：随机初始化生成的图像,通过在内容图像中添加随机噪声来产生噪声图像
    generated_image = nst_utils.generate_noise_image(content_image)
    imshow(generated_image[0])
    
    #第5步：加载VGG16模型
    model = nst_utils.load_vgg_model("E:\wuenda\pretrained-model/imagenet-vgg-verydeep-19.mat")
    #第6步：构建TensorFlow图：

    ##将内容图像作为VGG模型的输入。
    sess.run(model["input"].assign(content_image))
    
    ## 获取conv4_2层的输出
    out = model["conv4_2"]
    
    ## 将a_C设置为“conv4_2”隐藏层的激活值。
    a_C = sess.run(out)
    
    ## 将a_G设置为来自同一图层的隐藏层激活,这里a_G引用model["conv4_2"]，并且还没有计算，
    ## 在后面的代码中，我们将图像G指定为模型输入，这样当我们运行会话时，
    ## 这将是以图像G作为输入，从隐藏层中获取的激活值。
    a_G = out
    
    ## 计算内容成本
    J_content = compute_content_cost(a_C, a_G)
    
    ## 将风格图像作为VGG模型的输入
    sess.run(model["input"].assign(style_image))
    
    ## 计算风格成本
    J_style = compute_style_cost(model, STYLE_LAYERS)
    
    ## 计算总成本
    J = total_cost(J_content, J_style, alpha = 10, beta = 40)
    
    ## 定义优化器,设置学习率为2.0
    optimizer = tf.train.AdamOptimizer(2.0)
    
    ## 定义学习目标：最小化成本
    train_step = optimizer.minimize(J)
    #开始时间
    start_time = time.clock()
    
    #非GPU版本,约25-30min
    generated_image = model_nn(sess, generated_image)
    
    
    #使用GPU，约1-2min
    """
    with tf.device("/gpu:0"):
        generated_image = model_nn(sess, generated_image)
    """
    #结束时间
    end_time = time.clock()
    
    #计算时差
    minium = end_time - start_time

    print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium%60)) + "秒")





