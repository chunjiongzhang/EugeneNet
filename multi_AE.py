# seq2seq.py
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMCell
from keras.layers import Dense,LSTM, SimpleRNN, GRU,Input,Dropout,Activation
import keras
from keras.models import Model,Sequential
# from keras_radam import RAdam
# from mish import Mish  # mish 激活函数



#实验结果表明，双向的GRU的性能比双向的LSTM性能要好
#所以在skip_seq2seq.py中，使用GRU尝试一下

class MLP_AE:
    #传了tf占位符进来

    # def __init__(self, x,input_shape):
    #     self.x = x
    #     self.shape = input_shape
    #

    def encoder(x,shape):
        #encoder_op22 = tf.nn.l2_normalize(x, axis=1, epsilon=10e-5)
        #en0 = tf.layers.dense(x, shape, activation=tf.nn.relu)
        #en1 = tf.nn.l2_normalize(en0, axis=1, epsilon=10e-5)
        en2 = tf.layers.dense(x, 8, activation=tf.nn.tanh)
        #en1 = tf.nn.l2_normalize(en2, axis=1, epsilon=10e-5)
        # bn=tf.layers.batch_normalization(en0,training=is_train)
        # en1 = tf.layers.dense(en22, 28, activation=tf.nn.relu)
        # en2 = tf.layers.dense(en1, 28, tf.nn.relu)
        return en2

    def decoder(en2,shape):
        # decoder
        # de0 = tf.layers.dense(en2, 14, tf.nn.tanh)
        # de22 = tf.nn.l2_normalize(de0, axis=1, epsilon=10e-5)
        # de1 = tf.layers.dense(de22, 28, tf.nn.relu)
        de2 = tf.layers.dense(en2, 8, tf.nn.tanh)
        de3 = tf.layers.dense(de2, shape)
        return de3

class MLP_AE8:
    #传了tf占位符进来
    def encoder(x,shape):
        #encoder_op22 = tf.nn.l2_normalize(x, axis=1, epsilon=10e-5)
        en0 = tf.layers.dense(x, 60, activation=tf.nn.tanh)
        # bn=tf.layers.batch_normalization(en0,training=is_train)
        #en1 = tf.nn.l2_normalize(en0, axis=1, epsilon=10e-5)
        en1 = tf.layers.dense(en0, 30, activation=tf.nn.tanh)
        en2 = tf.layers.dense(en1, 10, tf.nn.tanh)
        return en2

    def decoder(en2,shape):
        # decoder
        de0 = tf.layers.dense(en2, 10, tf.nn.tanh)
        de1 = tf.layers.dense(de0, 30, tf.nn.tanh)
        de2 = tf.layers.dense(de1, 60, tf.nn.tanh)
        de3 = tf.layers.dense(de2, shape, activation=None)
        return de3


class MLP_AE2:
    #传了tf占位符进来
    def encoder(x,shape):
        #encoder_op22 = tf.nn.l2_normalize(x, axis=1, epsilon=10e-5)
        #en0 = tf.layers.dense(x, shape, activation=tf.nn.tanh)
        en0 = tf.layers.dense(x, 14, activation=tf.nn.tanh)
        en1 = tf.nn.l2_normalize(en0, axis=1, epsilon=10e-5)
        en2 = tf.layers.dense(en1, 28, activation=tf.nn.tanh)
        en3 = tf.layers.dense(en2, 28, activation=tf.nn.tanh)
        return en3
    def decoder(en2,shape):
        # decoder
        de0 = tf.layers.dense(en2, 14, tf.nn.tanh)
        de1 = tf.layers.dense(de0, 28, tf.nn.tanh)
        de2 = tf.layers.dense(de1, 28, tf.nn.tanh)
        de3 = tf.layers.dense(de2, shape)
        return de3

class MLP_AE4:
    #传了tf占位符进来

    def __init__(self, x,input_shape):
        self.x = x
        self.shape = input_shape
    def encoder(x,shape):
        #encoder_op22 = tf.nn.l2_normalize(x, axis=1, epsilon=10e-5)
        #en0 = tf.layers.dense(x, shape, activation=tf.nn.tanh)
        # bn=tf.layers.batch_normalization(en0,training=is_train)

        en1 = tf.layers.dense(x, 128, activation=tf.nn.tanh)
        en2 = tf.layers.dense(en1, 64, tf.nn.tanh)
        en2 = tf.layers.dense(en2, 32, tf.nn.tanh)
        return en2

    def decoder(en2,shape):
        # decoder
        de0 = tf.layers.dense(en2, 32, tf.nn.tanh)
        de1 = tf.layers.dense(de0, 64, tf.nn.tanh)
        de2 = tf.layers.dense(de1, 128, tf.nn.tanh)
        de3 = tf.layers.dense(de2, shape)
        return de3





class LSTM_AE:
    def __init__(self,x, input_shape):
        self.x = x
        self.shape = input_shape
    def encoder(data,shape):
        with tf.variable_scope("encoder", initializer=tf.orthogonal_initializer(), reuse=tf.AUTO_REUSE):
            data1= tf.reshape(data, (data.shape[0], 1, data.shape[1]))
            # 编码层
            lstmCell = tf.contrib.rnn.BasicLSTMCell(32)
            lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
            value, _ = tf.nn.dynamic_rnn(lstmCell, data1, dtype=tf.float32)
            # value=tf.layers.Flatten(value)
            value1 = tf.layers.dense(value, shape, activation=None)

        return value1

    def decoder(value1,shape):
        with tf.variable_scope("decoder", initializer=tf.orthogonal_initializer(), reuse=tf.AUTO_REUSE):
            lstmCell = tf.contrib.rnn.BasicLSTMCell(32)
            lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
            value1, _ = tf.nn.dynamic_rnn(lstmCell, value1, dtype=tf.float32)
            # value=tf.layers.Flatten(value)
            value = tf.layers.dense(value1, shape, activation=None)
        return value




class GRU_AE:
    #传了tf占位符进来
    def encoder(self,x):
        with tf.variable_scope("encoder", initializer=tf.orthogonal_initializer(), reuse=tf.AUTO_REUSE):
            previous_state = tf.Variable(tf.random_normal([20964, 4]))  # 前一个状态的输出
            gruCell = tf.nn.rnn_cell.GRUCell(4)
            output, state = tf.nn.dynamic_rnn(gruCell, x, time_major=False,initial_state=previous_state)
        return output,state

    def decoder(self,output,state):
        with tf.variable_scope("encoder", initializer=tf.orthogonal_initializer(), reuse=tf.AUTO_REUSE):
            previous_state = tf.Variable(tf.random_normal([20964, 4]))
            de3 = tf.layers.dense(de2, 67, activation=tf.nn.relu)# 前一个状态的输出
            gruCell = tf.nn.rnn_cell.GRUCell(4)
            output, state = tf.nn.dynamic_rnn(gruCell, output,initial_state=state, time_major=False)
            de2 = tf.layers.flatten(output)
            de3 = tf.layers.dense(de2, 67, activation=tf.nn.relu)
        return de3

class CNN_AE:
    def __init__(self, x,input_shape):
        self.x = x
        self.shape = input_shape
    def encoder(x,shape):
        #encoder_op22 = tf.nn.l2_normalize(x, axis=1, epsilon=10e-5)
        en0 = tf.layers.conv1d(inputs=x,filters=64,kernel_size=3,padding='valid',activation="relu")
        # bn=tf.layers.batch_normalization(en0,training=is_train)
        en1 = tf.layers.conv1d(en0, 64,3,activation=tf.nn.relu)
        en2 = tf.layers.max_pooling1d(inputs=en1,pool_size=2,strides=2,padding="SAME")
        en3 = tf.layers.conv1d(inputs=en2, filters=128, kernel_size=3, activation="relu")
        # bn=tf.layers.batch_normalization(en0,training=is_train)
        en4 = tf.layers.conv1d(en3, 128, 3, activation=tf.nn.relu)
        en5=tf.layers.max_pooling1d(en4,pool_size=2,strides=2,padding="SAME")
        en6=tf.layers.dense(en5,shape,activation=None)

        return en6
            
    def decoder(en6,shape):
        en0 = tf.layers.conv1d(inputs=en6, filters=64, kernel_size=3, padding='valid', activation="relu")
        # bn=tf.layers.batch_normalization(en0,training=is_train)
        en1 = tf.layers.conv1d(en0, 64, 3, activation=tf.nn.relu)
        en2 = tf.layers.max_pooling1d(inputs=en1, pool_size=2, strides=2, padding="SAME")
        en3 = tf.layers.conv1d(inputs=en2, filters=128, kernel_size=3, activation="relu")
        # bn=tf.layers.batch_normalization(en0,training=is_train)
        en4 = tf.layers.conv1d(en3, 128, 3, activation=tf.nn.relu)
        en5 = tf.layers.max_pooling1d(en4, pool_size=2, strides=2, padding="SAME")
        de6=tf.layers.flatten(en5)
        en7 = tf.layers.dense(de6, shape, activation=None)
        return en7
