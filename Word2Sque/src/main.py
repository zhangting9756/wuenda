#-*- coding: utf-8 -*-
import numpy as np
import w2v_utils
import emo_utils
np.random.seed(0)
import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
np.random.seed(1)
from keras.initializers import glorot_uniform
def cosine_similarity(u, v):
    """
    u与v的余弦相似度反映了u与v的相似程度
    
    参数：
        u -- 维度为(n,)的词向量
        v -- 维度为(n,)的词向量
        
    返回：
        cosine_similarity -- 由上面公式定义的u和v之间的余弦相似度。
    """
    distance = 0
    
    # 计算u与v的内积
    dot = np.dot(u, v)
    
    #计算u的L2范数
    norm_u = np.sqrt(np.sum(np.power(u, 2)))
    
    #计算v的L2范数
    norm_v = np.sqrt(np.sum(np.power(v, 2)))
    
    # 根据公式1计算余弦相似度
    cosine_similarity = np.divide(dot, norm_u * norm_v)
    
    return cosine_similarity
def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    解决“A与B相比就类似于C与____相比一样”之类的问题
    
    参数：
        word_a -- 一个字符串类型的词
        word_b -- 一个字符串类型的词
        word_c -- 一个字符串类型的词
        word_to_vec_map -- 字典类型，单词到GloVe向量的映射
        
    返回：
        best_word -- 满足(v_b - v_a) 最接近 (v_best_word - v_c) 的词
    """
    
    # 把单词转换为小写
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    
    # 获取对应单词的词向量
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
    
    # 获取全部的单词
    words = word_to_vec_map.keys()
    
    # 将max_cosine_sim初始化为一个比较大的负数
    max_cosine_sim = -100
    best_word = None
    
    # 遍历整个数据集
    for word in words:
        # 要避免匹配到输入的数据
        if word in [word_a, word_b, word_c]:
            continue
        # 计算余弦相似度
        cosine_sim = cosine_similarity((e_b - e_a), (word_to_vec_map[word] - e_c))
        
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = word
            
    return best_word
def sentence_to_avg(sentence, word_to_vec_map):
    """
    将句子转换为单词列表，提取其GloVe向量，然后将其平均。
    
    参数：
        sentence -- 字符串类型，从X中获取的样本。
        word_to_vec_map -- 字典类型，单词映射到50维的向量的字典
        
    返回：
        avg -- 对句子的均值编码，维度为(50,)
    """
    
    # 第一步：分割句子，转换为列表。
    words = sentence.lower().split()
    
    # 初始化均值词向量
    avg = np.zeros(50,)
    
    # 第二步：对词向量取平均。
    for w in words:
        avg += word_to_vec_map[w]
    avg = np.divide(avg, len(words))
    
    return avg

def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):
    """
    在numpy中训练词向量模型。
    
    参数：
        X -- 输入的字符串类型的数据，维度为(m, 1)。
        Y -- 对应的标签，0-7的数组，维度为(m, 1)。
        word_to_vec_map -- 字典类型的单词到50维词向量的映射。
        learning_rate -- 学习率.
        num_iterations -- 迭代次数。
        
    返回：
        pred -- 预测的向量，维度为(m, 1)。
        W -- 权重参数，维度为(n_y, n_h)。
        b -- 偏置参数，维度为(n_y,)
    """
    np.random.seed(1)
    
    # 定义训练数量
    m = Y.shape[0]
    n_y = 5
    n_h = 50
    
    # 使用Xavier初始化参数
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))
    
    # 将Y转换成独热编码
    Y_oh = emo_utils.convert_to_one_hot(Y, C=n_y)
    
    # 优化循环
    for t in range(num_iterations):
        for i in range(m):
            # 获取第i个训练样本的均值
            avg = sentence_to_avg(X[i], word_to_vec_map)
            
            # 前向传播
            z = np.dot(W, avg) + b
            a = emo_utils.softmax(z)
            
            # 计算第i个训练的损失
            cost = -np.sum(Y_oh[i]*np.log(a))
            
            # 计算梯度
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db = dz
            
            # 更新参数
            W = W - learning_rate * dW
            b = b - learning_rate * db
        if t % 100 == 0:
            print("第{t}轮，损失为{cost}".format(t=t,cost=cost))
            pred = emo_utils.predict(X, Y, W, b, word_to_vec_map)
            
    return pred, W, b

def sentences_to_indices(X, word_to_index, max_len):
    """
    输入的是X（字符串类型的句子的数组），再转化为对应的句子列表，
    输出的是能够让Embedding()函数接受的列表或矩阵（参见图4）。
    
    参数：
        X -- 句子数组，维度为(m, 1)
        word_to_index -- 字典类型的单词到索引的映射
        max_len -- 最大句子的长度，数据集中所有的句子的长度都不会超过它。
        
    返回：
        X_indices -- 对应于X中的单词索引数组，维度为(m, max_len)
    """
    
    m = X.shape[0]  # 训练集数量
    # 使用0初始化X_indices
    X_indices = np.zeros((m, max_len))
    
    for i in range(m):
        # 将第i个居住转化为小写并按单词分开。
        sentences_words = X[i].lower().split()
        
        # 初始化j为0
        j = 0
        
        # 遍历这个单词列表
        for w in sentences_words:
            # 将X_indices的第(i, j)号元素为对应的单词索引
            X_indices[i, j] = word_to_index[w]
            
            j += 1
            
    return X_indices

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    创建Keras Embedding()层，加载已经训练好了的50维GloVe向量
    
    参数：
        word_to_vec_map -- 字典类型的单词与词嵌入的映射
        word_to_index -- 字典类型的单词到词汇表（400,001个单词）的索引的映射。
        
    返回：
        embedding_layer() -- 训练好了的Keras的实体层。
    """
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["cucumber"].shape[0]
    
    # 初始化嵌入矩阵
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # 将嵌入矩阵的每行的“index”设置为词汇“index”的词向量表示
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    
    # 定义Keras的embbeding层
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    
    # 构建embedding层。
    embedding_layer.build((None,))
    
    # 将嵌入层的权重设置为嵌入矩阵。
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    """
    实现Emojify-V2模型的计算图
    
    参数：
        input_shape -- 输入的维度，通常是(max_len,)
        word_to_vec_map -- 字典类型的单词与词嵌入的映射。
        word_to_index -- 字典类型的单词到词汇表（400,001个单词）的索引的映射。
    
    返回：
        model -- Keras模型实体
    """
    # 定义sentence_indices为计算图的输入，维度为(input_shape,)，类型为dtype 'int32' 
    sentence_indices = Input(input_shape, dtype='int32')
    
    # 创建embedding层
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # 通过嵌入层传播sentence_indices，你会得到嵌入的结果
    embeddings = embedding_layer(sentence_indices)
    
    # 通过带有128维隐藏状态的LSTM层传播嵌入
    # 需要注意的是，返回的输出应该是一批序列。
    X = LSTM(128, return_sequences=True)(embeddings)
    # 使用dropout，概率为0.5
    X = Dropout(0.5)(X)
    # 通过另一个128维隐藏状态的LSTM层传播X
    # 注意，返回的输出应该是单个隐藏状态，而不是一组序列。
    X = LSTM(128, return_sequences=False)(X)
    # 使用dropout，概率为0.5
    X = Dropout(0.5)(X)
    # 通过softmax激活的Dense层传播X，得到一批5维向量。
    X = Dense(5)(X)
    # 添加softmax激活
    X = Activation('softmax')(X)
    
    # 创建模型实体
    model = Model(inputs=sentence_indices, outputs=X)
    
    return model


if __name__ == '__main__':
    """
    words, word_to_vec_map = w2v_utils.read_glove_vecs('F:\wuenda\data/glove.6B.50d.txt')
    #print(word_to_vec_map['orange'])
    father = word_to_vec_map["father"]
    mother = word_to_vec_map["mother"]
    ball = word_to_vec_map["ball"]
    crocodile = word_to_vec_map["crocodile"]
    france = word_to_vec_map["france"]
    italy = word_to_vec_map["italy"]
    paris = word_to_vec_map["paris"]
    rome = word_to_vec_map["rome"]
    
    print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
    print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile))
    print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy))

    triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
    for triad in triads_to_try:
        print ('{} -> {} <====> {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec_map)))
    """
    
    X_train, Y_train = emo_utils.read_csv('F:\wuenda/data/train_emoji.csv')
    X_test, Y_test = emo_utils.read_csv('F:\wuenda/data/test.csv')
    
    maxLen = len(max(X_train, key=len).split())
    index  = 3
    print(X_train[index], emo_utils.label_to_emoji(Y_train[index]))
    
    Y_oh_train = emo_utils.convert_to_one_hot(Y_train, C=5)
    Y_oh_test = emo_utils.convert_to_one_hot(Y_test, C=5)
    index = 0 
    print("{0}对应的独热编码是{1}".format(Y_train[index], Y_oh_train[index]))
    
    word_to_index, index_to_word, word_to_vec_map = emo_utils.read_glove_vecs('F:\wuenda/data/glove.6B.50d.txt')
    """
    word = "cucumber"
    index = 113317
    print("单词{0}对应的索引是：{1}".format(word, word_to_index[word]))
    print("索引{0}对应的单词是：{1}".format(index, index_to_word[index]))

    #avg = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vec_map)
    #print("avg = ", avg)
    
    print(X_train.shape)
    print(Y_train.shape)
    print(np.eye(5)[Y_train.reshape(-1)].shape)
    print(X_train[0])
    print(type(X_train))
    Y = np.asarray([5,0,0,5, 4, 4, 4, 6, 6, 4, 1, 1, 5, 6, 6, 3, 6, 3, 4, 4])
    print(Y.shape)
    
    X = np.asarray(['I am going to the bar tonight', 'I love you', 'miss you my dear',
     'Lets go party and drinks','Congrats on the new job','Congratulations',
     'I am so happy for you', 'Why are you feeling bad', 'What is wrong with you',
     'You totally deserve this prize', 'Let us go play football',
     'Are you down for football this afternoon', 'Work hard play harder',
     'It is suprising how people can be dumb sometimes',
     'I am very disappointed','It is the best day in my life',
     'I think I will end up alone','My life is so boring','Good job',
     'Great so awesome'])
    
    pred, W, b = model(X_train, Y_train, word_to_vec_map)
    print("=====训练集====")
    pred_train = emo_utils.predict(X_train, Y_train, W, b, word_to_vec_map)
    print("=====测试集====")
    pred_test = emo_utils.predict(X_test, Y_test, W, b, word_to_vec_map)

    print(" \t {0} \t {1} \t {2} \t {3} \t {4}".format(emo_utils.label_to_emoji(0), emo_utils.label_to_emoji(1), \
                                                     emo_utils.label_to_emoji(2), emo_utils.label_to_emoji(3), \
                                                     emo_utils.label_to_emoji(4)))
    import pandas as pd
    print(pd.crosstab(Y_test, pred_test.reshape(56,), rownames=['Actual'], colnames=['Predicted'], margins=True))
    emo_utils.plot_confusion_matrix(Y_test, pred_test)
    """
    X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
    X1_indices = sentences_to_indices(X1,word_to_index, max_len = 5)
    print("X1 =", X1)
    print("X1_indices =", X1_indices)

    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])

    model = Emojify_V2((10,), word_to_vec_map, word_to_index)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
    Y_train_oh = emo_utils.convert_to_one_hot(Y_train, C = 5)
    model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)

    X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
    Y_test_oh = emo_utils.convert_to_one_hot(Y_test, C = 5)
    loss, acc = model.evaluate(X_test_indices, Y_test_oh)
    
    print("Test accuracy = ", acc)
    
    C = 5
    y_test_oh = np.eye(C)[Y_test.reshape(-1)]
    X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
    pred = model.predict(X_test_indices)
    for i in range(len(X_test)):
        x = X_test_indices
        num = np.argmax(pred[i])
        if(num != Y_test[i]):
            print('正确表情：'+ emo_utils.label_to_emoji(Y_test[i]) + '   预测结果： '+ X_test[i] + emo_utils.label_to_emoji(num).strip())
    




