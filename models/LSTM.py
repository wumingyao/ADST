from keras.layers import Input, Dense, Activation, LSTM, Dropout
from keras.models import Model
from keras.layers.convolutional import Convolution1D
from models.iLayer import iLayer
import tensorflow as tf
from keras.layers import (
    Input,
    Activation,
    Dense,
    Reshape,
    Embedding,
    Add,
    Multiply,
    Concatenate
)
import config
import warnings

warnings.filterwarnings("ignore")
N_days = config.N_days  # 用了多少天的数据(目前17个工作日)
N_hours = config.N_hours
N_time_slice = config.N_time_slice  # 1小时有6个时间片
N_station = config.N_station  # 81个站点
N_flow = config.N_flow  # 进站 & 出站
len_seq1 = config.len_seq1  # week时间序列长度为2
len_seq2 = config.len_seq2  # day时间序列长度为3
len_seq3 = config.len_seq3  # hour时间序列长度为5
nb_flow = config.nb_flow  # 输入特征


# lstm
def LSTM_Net(c_conf=(len_seq3, N_flow, N_station), p_conf=(len_seq2, N_flow, N_station),
             t_conf=(len_seq1, N_flow, N_station),
             c1_conf=(len_seq3, N_station, N_station), p1_conf=(len_seq2, N_station, N_station),
             t1_conf=(len_seq1, N_station, N_station),
             external_dim1=None, external_dim2=None, external_dim3=None,
             external_dim4=None, external_dim5=None, external_dim6=None,
             external_dim7=None, external_dim8=None, external_dim9=None, nb_residual_unit=3, nb_edge_residual_unit=3):
    # main input
    main_inputs = []
    main_outputs = []
    outputs = []
    nb_flow = 2
    nb_stations = 81
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_feature, stations = conf
            input0 = Input(shape=(stations, nb_feature * len_seq))
            #            mask_target_input = Input(shape=(stations, nb_flow))
            main_inputs.append(input0)
            X0 = LSTM(512, activation='tanh', return_sequences=True)(input0)
            X0 = Dropout(0.5)(X0)
            X0 = Dense(81)(X0)
            X0 = Convolution1D(
                nb_filter=nb_flow, filter_length=3, border_mode="same")(X0)
            outputs.append(X0)

    for conf1 in [c1_conf, p1_conf, t1_conf]:
        if conf1 is not None:
            len_seq, nb_feature, stations = conf1
            input1 = Input(shape=(stations, nb_feature * len_seq))
            #            mask_target_input1 = Input(shape=(stations, stations))
            main_inputs.append(input1)
            #            main_inputs.append(mask_target_input1)
            X1 = LSTM(512, activation='tanh', return_sequences=True)(input1)
            X1 = Dense(81)(X1)
            X1 = Convolution1D(
                nb_filter=nb_flow, filter_length=3, border_mode="same")(X1)
            outputs.append(X1)

    mask_target_input1 = Input(shape=(N_station, N_station))
    main_inputs.append(mask_target_input1)
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
    conv_node = Convolution1D(nb_filter=nb_flow, filter_length=3, border_mode="same", name='node_logits_conv')(
        main_output)
    conv_edge = Convolution1D(nb_filter=nb_stations, filter_length=3, border_mode="same", name='edge_logits_conv')(
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

    if external_dim2 is not None and external_dim2 > 0:
        # external input 外部信息输入
        external_input2 = Input(shape=(external_dim2,))
        main_inputs.append(external_input2)
        embedding2 = Embedding(144, 20, input_length=1)(external_input2)
        # node 外部信息加入到node数据中
        h1_2 = Dense(output_dim=nb_flow * nb_stations)(embedding2)
        activation4_2 = Activation('relu')(h1_2)
        external_output1_2 = Reshape((nb_stations, nb_flow))(activation4_2)
        main_output1_2 = Add()([conv_node, external_output1_2])
        main_outputs[0] = main_output1_2
        # edge 外部信息加入到edge数据中
        h11_2 = Dense(output_dim=nb_stations * nb_stations)(embedding2)
        activation5_2 = Activation('relu')(h11_2)
        external_output11_2 = Reshape((nb_stations, nb_stations))(activation5_2)
        main_output2_2 = Add()([conv_edge, external_output11_2])
        main_outputs[1] = main_output2_2
    else:
        print('external_dim:', external_dim2)

    if external_dim4 is not None and external_dim4 > 0:
        # external input 外部信息输入
        external_input4 = Input(shape=(external_dim4,))
        main_inputs.append(external_input4)
        embedding4 = Embedding(2, 20, input_length=1)(external_input4)
        # node 外部信息加入到node数据中
        h1_4 = Dense(output_dim=nb_flow * nb_stations)(embedding4)
        activation4_4 = Activation('relu')(h1_4)
        external_output1_4 = Reshape((nb_stations, nb_flow))(activation4_4)
        main_output1_4 = Add()([conv_node, external_output1_4])
        main_outputs[0] = main_output1_4
        # edge 外部信息加入到edge数据中
        h11_4 = Dense(output_dim=nb_stations * nb_stations)(embedding4)
        activation5_4 = Activation('relu')(h11_4)
        external_output11_4 = Reshape((nb_stations, nb_stations))(activation5_4)
        main_output2_4 = Add()([conv_edge, external_output11_4])
        main_outputs[1] = main_output2_4
    else:
        print('external_dim:', external_dim4)

    # fusing with external component5
    if external_dim5 is not None and external_dim5 > 0:
        # external input
        external_input5 = Input(shape=(external_dim5,))
        main_inputs.append(external_input5)
        embedding5 = Dense(output_dim=20)(external_input5)
        embedding5 = Activation('relu')(embedding5)
        # node 外部信息加入到node数据中
        h1_5 = Dense(output_dim=nb_flow * nb_stations)(embedding5)
        activation4_5 = Activation('relu')(h1_5)
        external_output1_5 = Reshape((nb_stations, nb_flow))(activation4_5)
        main_output1_5 = Add()([conv_node, external_output1_5])
        main_outputs[0] = main_output1_5
        # edge 外部信息加入到edge数据中
        h11_5 = Dense(output_dim=nb_stations * nb_stations)(embedding5)
        activation5_5 = Activation('relu')(h11_5)
        external_output11_5 = Reshape((nb_stations, nb_stations))(activation5_5)
        main_output2_5 = Add()([conv_edge, external_output11_5])
        main_outputs[1] = main_output2_5
    else:
        print('external_dim:', external_dim5)

    if external_dim9 is not None and external_dim9 > 0:
        # external input 外部信息输入
        external_input9 = Input(shape=(external_dim9,))
        main_inputs.append(external_input9)
        embedding9 = Embedding(3, 20, input_length=1)(external_input9)
        # node 外部信息加入到node数据中
        h1_9 = Dense(output_dim=nb_flow * nb_stations)(embedding9)
        activation4_9 = Activation('relu')(h1_9)
        external_output1_9 = Reshape((nb_stations, nb_flow))(activation4_9)
        main_output1_9 = Add(name='node_logits')([conv_node, external_output1_9])  # 改了这里
        main_outputs[0] = main_output1_9
        # edge 外部信息加入到edge数据中
        h11_9 = Dense(output_dim=nb_stations * nb_stations)(embedding9)
        activation5_9 = Activation('relu')(h11_9)
        external_output11_9 = Reshape((nb_stations, nb_stations))(activation5_9)
        main_output2_9 = Add()([conv_edge, external_output11_9])
        main_outputs[1] = main_output2_9
    else:
        print('external_dim:', external_dim9)

    # masking node/edge
    # 对node数据和edge数据进行mask操作，实现损失函数中要求的输出
    #    mask_node = Multiply(name="node_logits")([mask_target_input, main_outputs[0]])
    mask_edge = Multiply(name="edge_logits")([mask_target_input1, main_outputs[1]])

    #    main_outputs[0] = mask_node
    main_outputs[1] = mask_edge

    # 完成模型搭建
    # 输入为：node数据、edge数据、node_mask、edge_mask，
    # 输出为：预测的masking_node数据、masking_edge数据

    model = Model(main_inputs, main_outputs)
    return model
