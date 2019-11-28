'''
    ADST-Net: Attention based Deep Spatio-temporal Residual Network
'''
from __future__ import print_function
import config

N_days = 160  # 用了多少天的数据(目前17个工作日)
N_hours = config.N_hours
N_time_slice = 2  # 1小时有6个时间片
N_station = 81  # 81个站点
N_flow = config.N_flow  # 进站 & 出站
len_seq1 = config.len_seq1  # week时间序列长度为2
len_seq2 = config.len_seq2  # day时间序列长度为3
len_seq3 = config.len_seq3  # hour时间序列长度为5
nb_flow = config.nb_flow  # 输入特征
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from keras.layers import (
    Input,
    Activation,
    Dense,
    Reshape,
    GlobalAveragePooling1D,
    multiply,
    Multiply,
    Embedding,
    Flatten,
    Dropout,
    LSTM,
    Add,
    Concatenate
)
from keras.layers.convolutional import Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
# from keras.utils.visualize_util import plot
from models.iLayer import iLayer
import keras.backend as K
import keras.layers as KL


# SE块--全局池化再重新赋权重，实现Attention注意力机制
def se_block(block_input, num_filters, ratio=18):  # Squeeze and excitation block
    pool1 = GlobalAveragePooling1D()(block_input)
    flat = Reshape((1, num_filters))(pool1)
    dense1 = Dense(num_filters // ratio, activation='relu')(flat)
    dense2 = Dense(num_filters, activation='sigmoid')(dense1)
    scale = multiply([block_input, dense2])
    return scale


# 残差连接
def _shortcut(input, residual):
    return Add()([input, residual])


# BN层+ReLU层+卷积层，整合成一个块 (没有用BN)
def _bn_relu_conv(nb_filter, bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Convolution1D(nb_filter=nb_filter, filter_length=3, border_mode="same")(activation)

    return f


# 残差单元，包含两个BN_ReLU_Conv块，以及一个SE块
def _residual_unit(nb_filter):
    def f(input):
        residual = _bn_relu_conv(nb_filter)(input)
        residual = _bn_relu_conv(nb_filter)(residual)
        se = se_block(residual, num_filters=nb_filter)
        return _shortcut(input, se)

    return f


# 残差网络--包含repetition个残差单元，并在残差单元之间插入SE块 repetition超参数
def ResUnits(residual_unit, nb_filter, repetations=1):
    def f(input):
        for i in range(repetations):
            input = residual_unit(nb_filter=nb_filter)(input)
            se = se_block(input, num_filters=nb_filter)
        return se

    return f


# ST-ResNet网络
def lstm_TaxiBJ(c_conf=(len_seq3, N_flow, N_station), p_conf=(len_seq2, N_flow, N_station),
                    t_conf=(len_seq1, N_flow, N_station)):
    # main input
    main_inputs = []
    main_outputs = []
    outputs = []
    nb_flow = 2
    # 针对C、P、T三种时间范围的Node数据进行卷积
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_feature, stations = conf
            input0 = Input(shape=(stations, nb_feature * len_seq))
            #            mask_target_input = Input(shape=(stations, nb_flow))
            main_inputs.append(input0)
            #            main_inputs.append(mask_target_input)
            X0 = LSTM(512, activation='tanh', return_sequences=True)(input0)
            X0 = Dropout(0.5)(X0)
            X0 = Dense(81)(X0)
            X0 = Convolution1D(
                nb_filter=nb_flow, filter_length=3, border_mode="same")(X0)
            outputs.append(X0)
    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        # Bridge操作，即对Node数据和Edge数据进行简单地concatenate操作
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output))
        # 主任务和附任务连接
        main_output = Concatenate()(new_outputs)

    # 对Bridge数据进行不同维度地卷积，实现两个不同任务的输出
    # 不同维度的输出
    conv_node = Convolution1D(nb_filter=nb_flow, filter_length=3, border_mode="same", name='node_logits')(main_output)

    main_outputs.append(conv_node)
    # fusing node/edge with external component1
    # 将外部信息分别整合到Node数据和Edge数据中

    # 完成模型搭建
    # 输入为：node数据、edge数据、node_mask、edge_mask，
    # 输出为：预测的masking_node数据、masking_edge数据

    model = Model(main_inputs, main_outputs)
    return model
