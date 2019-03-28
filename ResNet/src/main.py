# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import kt_utils
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,Add
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import kt_utils 
from keras.initializers import glorot_uniform

from IPython.display import SVG
import scipy.misc
from matplotlib.pyplot import imshow
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def model(input_shape):
    """
    模型大纲
    """
    #定义一个tensor的placeholder，维度为input_shape
    X_input = Input(input_shape)

    #使用0填充：X_input的周围填充0
    X = ZeroPadding2D((3,3))(X_input)

    # 对X使用 CONV -> BN -> RELU 块
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    #最大值池化层
    X = MaxPooling2D((2,2),name="max_pool")(X)

    #降维，矩阵转化为向量 + 全连接层
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    #创建模型，讲话创建一个模型的实体，我们可以用它来训练、测试。
    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    return model

def HappyModel(input_shape):
    """
    实现一个检测笑容的模型

    参数：
        input_shape - 输入的数据的维度
    返回：
        model - 创建的Keras的模型

    """

    #你可以参考和上面的大纲
    X_input = Input(input_shape)

    #使用0填充：X_input的周围填充0
    X = ZeroPadding2D((3, 3))(X_input)

    #对X使用 CONV -> BN -> RELU 块
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    #最大值池化层
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    #降维，矩阵转化为向量 + 全连接层
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    #创建模型，讲话创建一个模型的实体，我们可以用它来训练、测试。
    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model

def identity_block(X, f, filters, stage, block):
    """
    实现图3的恒等块

    参数：
        X - 输入的tensor类型的数据，维度为( m, n_H_prev, n_W_prev, n_H_prev )
        f - 整数，指定主路径中间的CONV窗口的维度
        filters - 整数列表，定义了主路径每层的卷积层的过滤器数量
        stage - 整数，根据每层的位置来命名每一层，与block参数一起使用。
        block - 字符串，据每层的位置来命名每一层，与stage参数一起使用。

    返回：
        X - 恒等块的输出，tensor类型，维度为(n_H, n_W, n_C)

    """

    #定义命名规则
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base   = "bn"  + str(stage) + block + "_branch"

    #获取过滤器
    F1, F2, F3 = filters

    #保存输入数据，将会用于为主路径添加捷径
    X_shortcut = X

    #主路径的第一部分
    ##卷积层
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1) ,padding="valid",
               name=conv_name_base+"2a", kernel_initializer=glorot_uniform(seed=0))(X)
    ##归一化
    X = BatchNormalization(axis=3,name=bn_name_base+"2a")(X)
    ##使用ReLU激活函数
    X = Activation("relu")(X)

    #主路径的第二部分
    ##卷积层
    X = Conv2D(filters=F2, kernel_size=(f,f),strides=(1,1), padding="same",
               name=conv_name_base+"2b", kernel_initializer=glorot_uniform(seed=0))(X)
    ##归一化
    X = BatchNormalization(axis=3,name=bn_name_base+"2b")(X)
    ##使用ReLU激活函数
    X = Activation("relu")(X)


    #主路径的第三部分
    ##卷积层
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding="valid",
               name=conv_name_base+"2c", kernel_initializer=glorot_uniform(seed=0))(X)
    ##归一化
    X = BatchNormalization(axis=3,name=bn_name_base+"2c")(X)
    ##没有ReLU激活函数

    #最后一步：
    ##将捷径与输入加在一起
    X = Add()([X,X_shortcut])
    ##使用ReLU激活函数
    X = Activation("relu")(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):
    """
    实现图5的卷积块

    参数：
        X - 输入的tensor类型的变量，维度为( m, n_H_prev, n_W_prev, n_C_prev)
        f - 整数，指定主路径中间的CONV窗口的维度
        filters - 整数列表，定义了主路径每层的卷积层的过滤器数量
        stage - 整数，根据每层的位置来命名每一层，与block参数一起使用。
        block - 字符串，据每层的位置来命名每一层，与stage参数一起使用。
        s - 整数，指定要使用的步幅

    返回：
        X - 卷积块的输出，tensor类型，维度为(n_H, n_W, n_C)
    """

    #定义命名规则
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base   = "bn"  + str(stage) + block + "_branch"

    #获取过滤器数量
    F1, F2, F3 = filters

    #保存输入数据
    X_shortcut = X

    #主路径
    ##主路径第一部分
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(s,s), padding="valid",
               name=conv_name_base+"2a", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2a")(X)
    X = Activation("relu")(X)

    ##主路径第二部分
    X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding="same",
               name=conv_name_base+"2b", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2b")(X)
    X = Activation("relu")(X)

    ##主路径第三部分
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding="valid",
               name=conv_name_base+"2c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2c")(X)

    #捷径
    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding="valid",
               name=conv_name_base+"1", kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3,name=bn_name_base+"1")(X_shortcut)

    #最后一步
    X = Add()([X,X_shortcut])
    X = Activation("relu")(X)

    return X


if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = kt_utils.load_dataset()
    
    # Normalize image vectors
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    
    # Reshape
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T
    index = 3
    plt.imshow(X_train_orig[index])
    plt.show()
    """
    #创建一个模型实体
    happy_model = HappyModel(X_train.shape[1:])
    #编译模型
    happy_model.compile("adam","binary_crossentropy", metrics=['accuracy'])
    #训练模型
    #请注意，此操作会花费你大约6-10分钟。
    happy_model.fit(X_train, Y_train, epochs=40, batch_size=50)
    #评估模型
    preds = happy_model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)
    print ("误差值 = " + str(preds[0]))
    print ("准确度 = " + str(preds[1]))
    """
    tf.reset_default_graph()
    with tf.Session() as test:
        np.random.seed(1)
        A_prev = tf.placeholder("float",[3,4,4,6])
        X = np.random.randn(3,4,4,6)
        A = identity_block(A_prev,f=2,filters=[2,4,6],stage=1,block="a")
    
        test.run(tf.global_variables_initializer())
        out = test.run([A],feed_dict={A_prev:X,K.learning_phase():0})
        print("out = " + str(out[0][1][1][0]))
    
        test.close()
        
    tf.reset_default_graph()

    with tf.Session() as test:
        np.random.seed(1)
        A_prev = tf.placeholder("float",[3,4,4,6])
        X = np.random.randn(3,4,4,6)
    
        A = convolutional_block(A_prev,f=2,filters=[2,4,6],stage=1,block="a")
        test.run(tf.global_variables_initializer())
    
        out = test.run([A],feed_dict={A_prev:X,K.learning_phase():0})
        print("out = " + str(out[0][1][1][0]))
    
        test.close()


