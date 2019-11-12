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
    multiply,
    Multiply,
    Embedding,
    Flatten,
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
import warnings

warnings.filterwarnings("ignore")


# SE块--全局池化再重新赋权重，实现注意力机制
def se_block(block_input, num_filters, ratio=8):  # Squeeze and excitation block
    pool1 = GlobalAveragePooling1D()(block_input)
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


# 残差单元，包含两个BN_ReLU_Conv块，以及一个SE快
def _residual_unit(nb_filter):
    def f(input):
        residual = _bn_relu_conv(nb_filter)(input)
        residual = _bn_relu_conv(nb_filter)(residual)
        se = se_block(residual, num_filters=nb_filter)
        return _shortcut(input, se)

    return f


# 残差网络--包含repetition个残差单元，并在残差单元之间插入SE块
def ResUnits(residual_unit, nb_filter, repetations=1):
    def f(input):
        for i in range(repetations):
            input = residual_unit(nb_filter=nb_filter)(input)
            se = se_block(input, num_filters=nb_filter)
        return se

    return f


# ST-ResNet网络
def stresnet(c_conf=(3, 2, 81), p_conf=(3, 2, 81), t_conf=(3, 2, 81),
             c1_conf=(3, 81, 81), p1_conf=(3, 81, 81), t1_conf=(3, 81, 81),
             external_dim1=None, nb_residual_unit=3):
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
            #            mask_target_input = Input(shape=(stations, nb_flow))
            main_inputs.append(input0)
            #            main_inputs.append(mask_target_input)
            # Conv1
            conv1 = Convolution1D(
                nb_filter=64, filter_length=3, border_mode="same")(input0)
            # [nb_residual_unit] Residual Units
            residual_output = ResUnits(_residual_unit, nb_filter=64,
                                       repetations=nb_residual_unit)(conv1)
            # Conv2
            activation = Activation('relu')(residual_output)
            conv2 = Convolution1D(
                nb_filter=nb_flow, filter_length=3, border_mode="same")(activation)
            outputs.append(conv2)

    # 针对C、P、T三种时间范围的Edge数据进行卷积
    for conf1 in [c1_conf, p1_conf, t1_conf]:
        if conf1 is not None:
            len_seq, nb_feature, stations = conf1
            input1 = Input(shape=(stations, nb_feature * len_seq))
            #            mask_target_input1 = Input(shape=(stations, stations))
            main_inputs.append(input1)
            #            main_inputs.append(mask_target_input1)
            # Conv1
            conv11 = Convolution1D(
                nb_filter=32, filter_length=3, border_mode="same")(input1)
            # [nb_residual_unit] Residual Units
            residual_output1 = ResUnits(_residual_unit, nb_filter=32, repetations=nb_residual_unit)(conv11)
            # Conv2
            activation1 = Activation('relu')(residual_output1)
            conv21 = Convolution1D(
                nb_filter=stations, filter_length=3, border_mode="same")(activation1)
            outputs.append(conv21)

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
    conv_edge = Convolution1D(nb_filter=nb_stations, filter_length=3, border_mode="same", name='edge_logits')(
        main_output)

    main_outputs.append(conv_node)
    main_outputs.append(conv_edge)
    # fusing node/edge with external component1
    # 将外部信息分别整合到Node数据和Edge数据中
    if external_dim1 is not None and external_dim1 > 0:
        # external input 外部信息输入
        external_input1 = Input(shape=(external_dim1,))
        main_inputs.append(external_input1)
        embedding1 = Dense(output_dim=20)(external_input1)
        embedding1 = Activation('relu')(embedding1)
        # node 外部信息加入到node数据中
        h1 = Dense(output_dim=nb_flow * nb_stations)(embedding1)
        activation1 = Activation('relu')(h1)
        external_output1 = Reshape((nb_stations, nb_flow))(activation1)
        main_output1 = Add()([conv_node, external_output1])
        main_output1 = Activation('tanh')(main_output1)
        main_outputs[0] = main_output1
        # edge 外部信息加入到edge数据中
        h11 = Dense(output_dim=nb_stations * nb_stations)(embedding1)
        activation11 = Activation('relu')(h11)
        external_output11 = Reshape((nb_stations, nb_stations))(activation11)
        main_output2 = Add()([conv_edge, external_output11])
        main_output2 = Activation('tanh')(main_output2)
        main_outputs[1] = main_output2
    else:
        print('external_dim:', external_dim1)

    # masking node/edge
    # 对node数据和edge数据进行mask操作，实现损失函数中要求的输出
    #    mask_node = Multiply(name="node_logits")([mask_target_input, main_outputs[0]])
    #    mask_edge = Multiply(name="edge_logits")([mask_target_input1, main_outputs[1]])

    #    main_outputs[0] = mask_node
    #    main_outputs[1] = mask_edge

    # 完成模型搭建
    # 输入为：node数据、edge数据、node_mask、edge_mask，
    # 输出为：预测的masking_node数据、masking_edge数据
    model = Model(main_inputs, main_outputs)
    return model
