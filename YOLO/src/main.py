# -*- coding: UTF-8 -*-
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model

from keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
import yolo_utils
def yolo_filter_boxes(box_confidence , boxes, box_class_probs, threshold = 0.6):
    """
    通过阈值来过滤对象和分类的置信度。

    参数：
        box_confidence  - tensor类型，维度为（19,19,5,1）,包含19x19单元格中每个单元格预测的5个锚框中的所有的锚框的pc （一些对象的置信概率）。
        boxes - tensor类型，维度为(19,19,5,4)，包含了所有的锚框的（px,py,ph,pw ）。
        box_class_probs - tensor类型，维度为(19,19,5,80)，包含了所有单元格中所有锚框的所有对象( c1,c2,c3，···，c80 )检测的概率。
        threshold - 实数，阈值，如果分类预测的概率高于它，那么这个分类预测的概率就会被保留。

    返回：
        scores - tensor 类型，维度为(None,)，包含了保留了的锚框的分类概率。
        boxes - tensor 类型，维度为(None,4)，包含了保留了的锚框的(b_x, b_y, b_h, b_w)
        classess - tensor 类型，维度为(None,)，包含了保留了的锚框的索引

    注意："None"是因为你不知道所选框的确切数量，因为它取决于阈值。
          比如：如果有10个锚框，scores的实际输出大小将是（10,）
    """

    #第一步：计算锚框的得分
    box_scores  = box_confidence * box_class_probs

    #第二步：找到最大值的锚框的索引以及对应的最大值的锚框的分数
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)

    #第三步：根据阈值创建掩码
    filtering_mask = (box_class_scores >= threshold)

    #对scores, boxes 以及 classes使用掩码
    scores = tf.boolean_mask(box_class_scores,filtering_mask)
    boxes = tf.boolean_mask(boxes,filtering_mask)
    classes = tf.boolean_mask(box_classes,filtering_mask)

    return scores , boxes , classes

def iou(box1, box2):
    """
    实现两个锚框的交并比的计算

    参数：
        box1 - 第一个锚框，元组类型，(x1, y1, x2, y2)
        box2 - 第二个锚框，元组类型，(x1, y1, x2, y2)

    返回：
        iou - 实数，交并比。
    """
    #计算相交的区域的面积
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1])
    xi2 = np.minimum(box1[2], box2[2])
    yi2 = np.minimum(box1[3], box2[3])
    inter_area = (xi1-xi2)*(yi1-yi2)

    #计算并集，公式为：Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area

    #计算交并比
    iou = inter_area / union_area

    return iou

def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    为锚框实现非最大值抑制（ Non-max suppression (NMS)）

    参数：
        scores - tensor类型，维度为(None,)，yolo_filter_boxes()的输出
        boxes - tensor类型，维度为(None,4)，yolo_filter_boxes()的输出，已缩放到图像大小（见下文）
        classes - tensor类型，维度为(None,)，yolo_filter_boxes()的输出
        max_boxes - 整数，预测的锚框数量的最大值
        iou_threshold - 实数，交并比阈值。

    返回：
        scores - tensor类型，维度为(,None)，每个锚框的预测的可能值
        boxes - tensor类型，维度为(4,None)，预测的锚框的坐标
        classes - tensor类型，维度为(,None)，每个锚框的预测的分类

    注意："None"是明显小于max_boxes的，这个函数也会改变scores、boxes、classes的维度，这会为下一步操作提供方便。

    """
    max_boxes_tensor = K.variable(max_boxes,dtype="int32") #用于tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) #初始化变量max_boxes_tensor

    #使用使用tf.image.non_max_suppression()来获取与我们保留的框相对应的索引列表
    nms_indices = tf.image.non_max_suppression(boxes, scores,max_boxes,iou_threshold)

    #使用K.gather()来选择保留的锚框
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes
def yolo_eval(yolo_outputs, image_shape=(720.,1280.), 
              max_boxes=10, score_threshold=0.6,iou_threshold=0.5):
    """
    将YOLO编码的输出（很多锚框）转换为预测框以及它们的分数，框坐标和类。

    参数：
        yolo_outputs - 编码模型的输出（对于维度为（608,608,3）的图片），包含4个tensors类型的变量：
                        box_confidence ： tensor类型，维度为(None, 19, 19, 5, 1)
                        box_xy         ： tensor类型，维度为(None, 19, 19, 5, 2)
                        box_wh         ： tensor类型，维度为(None, 19, 19, 5, 2)
                        box_class_probs： tensor类型，维度为(None, 19, 19, 5, 80)
        image_shape - tensor类型，维度为（2,），包含了输入的图像的维度，这里是(608.,608.)
        max_boxes - 整数，预测的锚框数量的最大值
        score_threshold - 实数，可能性阈值。
        iou_threshold - 实数，交并比阈值。

    返回：
        scores - tensor类型，维度为(,None)，每个锚框的预测的可能值
        boxes - tensor类型，维度为(4,None)，预测的锚框的坐标
        classes - tensor类型，维度为(,None)，每个锚框的预测的分类
    """

    #获取YOLO模型的输出
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    #中心点转换为边角
    boxes = yolo_boxes_to_corners(box_xy,box_wh)

    #可信度分值过滤
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    #缩放锚框，以适应原始图像
    boxes = yolo_utils.scale_boxes(boxes, image_shape)

    #使用非最大值抑制
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes
def leaky_relu(x, leak=0.2, name="LeakyRelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * tf.abs(x) 
if __name__ == '__main__':
    with tf.Session() as test_a:
        box_confidence = tf.random_normal([19,19,5,1], mean=1, stddev=4, seed=1)
        boxes = tf.random_normal([19,19,5,4],  mean=1, stddev=4, seed=1)
        box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
        scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
    
        print("scores[2] = " + str(scores[2].eval()))
        print("boxes[2] = " + str(boxes[2].eval()))
        print("classes[2] = " + str(classes[2].eval()))
        print("scores.shape = " + str(scores.shape))
        print("boxes.shape = " + str(boxes.shape))
        print("classes.shape = " + str(classes.shape))
    
        test_a.close()
    box1 = (2,1,4,3)
    box2 = (1,2,3,4)

    print("iou = " + str(iou(box1, box2)))
    with tf.Session() as test_b:
        scores = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
        boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed = 1)
        classes = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
        scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
    
        print("scores[2] = " + str(scores[2].eval()))
        print("boxes[2] = " + str(boxes[2].eval()))
        print("classes[2] = " + str(classes[2].eval()))
        print("scores.shape = " + str(scores.eval().shape))
        print("boxes.shape = " + str(boxes.eval().shape))
        print("classes.shape = " + str(classes.eval().shape))
    
        test_b.close()
        
    with tf.Session() as test_c:
        yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                        tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                        tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                        tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
        scores, boxes, classes = yolo_eval(yolo_outputs)
    
        print("scores[2] = " + str(scores[2].eval()))
        print("boxes[2] = " + str(boxes[2].eval()))
        print("classes[2] = " + str(classes[2].eval()))
        print("scores.shape = " + str(scores.eval().shape))
        print("boxes.shape = " + str(boxes.eval().shape))
        print("classes.shape = " + str(classes.eval().shape))
    
        test_c.close()

    sess = K.get_session()
    class_names = yolo_utils.read_classes("E:\wuenda\model_data/coco_classes.txt")
    anchors = yolo_utils.read_anchors("E:\wuenda\model_data/yolo_anchors.txt")
    image_shape = (720.,1280.)
    print("ztbefore")
    yolo_model = load_model("E:\wuenda\model_data/yolov2.h5")
    print("zt")
    yolo_model.summary()



