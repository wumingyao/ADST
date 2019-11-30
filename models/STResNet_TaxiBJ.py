'''
    ST-ResNet: Deep Spatio-temporal Residual Networks
'''

from __future__ import print_function
import numpy as np
from keras.layers import (
    Input,
    Activation,
    Dense,
    Reshape,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    multiply,
    Multiply,
    Embedding,
    Flatten,
    Add,
    Concatenate
)
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
# from keras.utils.visualize_util import plot
from models.iLayer import iLayer
import keras.backend as K
import keras.layers as KL


# SE块--全局池化再重新赋权重，实现注意力机制
def se_block(block_input, num_filters, ratio=8):  # Squeeze and excitation block
    pool1 = GlobalAveragePooling1D()(block_input)
    flat = Reshape((1, num_filters))(pool1)
    dense1 = Dense(num_filters // ratio, activation='relu')(flat)
    dense2 = Dense(num_filters, activation='sigmoid')(dense1)
    scale = multiply([block_input, dense2])

    return scale


def se_block2d(block_input, num_filters, ratio=8):  # Squeeze and excitation block
    pool1 = GlobalAveragePooling2D()(block_input)
    flat = Reshape((1, num_filters))(pool1)
    dense1 = Dense(num_filters // ratio, activation='relu')(flat)
    dense2 = Dense(num_filters, activation='sigmoid')(dense1)
    scale = multiply([block_input, dense2])

    return scale


def _shortcut(input, residual):
    return Add()([input, residual])


# BN层+ReLU层+卷积层，整合成一个块
def _bn_relu_conv(nb_filter, bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Convolution1D(nb_filter=nb_filter, filter_length=3, border_mode="same")(activation)

    return f


# BN层+ReLU层+卷积层，整合成一个块
def _bn_relu_conv2d(nb_filter, bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Convolution2D(filters=nb_filter, kernel_size=3, border_mode="same")(activation)

    return f


# 残差单元，包含两个BN_ReLU_Conv块，以及一个SE快
def _residual_unit(nb_filter):
    def f(input):
        residual = _bn_relu_conv(nb_filter)(input)
        residual = _bn_relu_conv(nb_filter)(residual)
        se = se_block(residual, num_filters=nb_filter)
        return _shortcut(input, se)

    return f


def _residual_unit_2d(nb_filter):
    def f(input):
        residual = _bn_relu_conv2d(nb_filter)(input)
        residual = _bn_relu_conv2d(nb_filter)(residual)
        se = se_block2d(residual, num_filters=nb_filter)
        return _shortcut(input, se)

    return f


# 残差网络--包含repetations个残差单元，并在残差单元之间插入SE块
def ResUnits(residual_unit, se_unit, nb_filter, repetations=1):
    def f(input):
        for i in range(repetations):
            input = residual_unit(nb_filter=nb_filter)(input)
            se = se_unit(input, num_filters=nb_filter)
        return se

    return f


# ST-ResNet网络
def stresnet_TaxiBJ_2D(c_conf=(3, 2, 81), p_conf=(3, 2, 81), t_conf=(3, 2, 81), nb_residual_unit=3):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (len_seq, nb_flow, Metro_stations)
    external_dim为外部信息维度
    '''
    # main input
    main_inputs = []
    main_outputs = []
    outputs = []
    nb_flow = 2
    nb_stations = 81
    # 针对C、P、T三种时间范围的Node数据进行卷积
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_feature, stations = conf
            input0 = Input(shape=(stations, nb_feature * len_seq))
            main_inputs.append(input0)
            # Conv1
            conv1 = Convolution1D(
                nb_filter=64, filter_length=3, border_mode="same")(input0)
            # [nb_residual_unit] Residual Units
            residual_output = ResUnits(_residual_unit, se_block, nb_filter=64,
                                       repetations=nb_residual_unit)(conv1)
            # Conv2
            activation = Activation('relu')(residual_output)
            conv2 = Convolution1D(
                nb_filter=nb_flow, filter_length=3, border_mode="same")(activation)
            outputs.append(conv2)

    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        # Bridge操作，即对Node数据和Edge数据进行简单地concatenate操作
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output))
        main_output = Concatenate()(new_outputs)

    # 对Bridge数据进行不同维度地卷积，实现两个不同任务的输出
    conv_node = Convolution1D(nb_filter=nb_flow, filter_length=3, border_mode="same", name='node_logits')(main_output)

    main_outputs.append(conv_node)

    # 完成模型搭建
    # 输入为：node数据、edge数据、node_mask、edge_mask，
    # 输出为：预测的masking_node数据、masking_edge数据
    model = Model(main_inputs, main_outputs)
    return model
