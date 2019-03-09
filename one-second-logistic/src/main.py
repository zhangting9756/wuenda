# -*- coding: UTF-8 -*-
import numpy as np
import h5py
import lr_utils
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = lr_utils.load_dataset()
    index = 25
    plt.imshow(train_set_x_orig[index])
    plt.show()
    print("y=" + str(train_set_y[:,index]) + ", it's a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + "' picture")
    
    m_train = train_set_y.shape[1] #训练集里图片的数量。
    m_test = test_set_y.shape[1] #测试集里图片的数量。
    num_px = train_set_x_orig.shape[1] #训练、测试集里面的图片的宽度和高度（均为64x64）。
    """
    print ("训练集的数量: m_train = " + str(m_train))
    print ("测试集的数量 : m_test = " + str(m_test))
    print ("每张图片的宽/高 : num_px = " + str(num_px))
    print ("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("训练集_图片的维数 : " + str(train_set_x_orig.shape))
    print ("训练集_标签的维数 : " + str(train_set_y.shape))
    print ("测试集_图片的维数: " + str(test_set_x_orig.shape))
    print ("测试集_标签的维数: " + str(test_set_y.shape))
    """
    #X_flatten = X.reshape(X.shape [0]，-1).T ＃X.T是X的转置
    #将训练集的维度降低并转置。
    train_set_x_flatten  = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
    #将测试集的维度降低并转置。
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    """
    print ("训练集降维最后的维度： " + str(train_set_x_flatten.shape))
    print ("训练集_标签的维数 : " + str(train_set_y.shape))
    print ("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
    print ("测试集_标签的维数 : " + str(test_set_y.shape))
    """
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255

    print("====================测试optimize====================")
    w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
    params , grads , costs = lr_utils.optimize(w , b , X , Y , num_iterations=100 , learning_rate = 0.009 , print_cost = False)
    print ("w = " + str(params["w"]))
    print ("b = " + str(params["b"]))
    print ("dw = " + str(grads["dw"]))
    print ("db = " + str(grads["db"]))
    
    print("====================测试predict====================")
    w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
    print("predictions = " + str(lr_utils.predict(w, b, X)))
    print("====================测试model====================")     
    #这里加载的是真实的数据，请参见上面的代码部分。
    d = lr_utils.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.95, print_cost = True)

