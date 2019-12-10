from keras.layers import (
    Input,
    Activation,
    merge,
    Add,
    Dense,
    Reshape,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    multiply,
    Permute,
    Concatenate
)
from models.iLayer import iLayer
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def _shortcut(input, residual):
    # residual = SE(residual, ratio=32, channel=64)
    return Add()([input, residual])


def _bn_relu_conv(nb_filter, nb_row, nb_col, bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, border_mode="same",
                             data_format='channels_first')(activation)
        # conv = SE(conv, ratio=32, channel=64)
        return conv

    return f


def _residual_unit(nb_filter):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return _shortcut(input, residual)

    return f


def ResUnits(residual_unit, nb_filter, repetations=5):
    def f(input):
        for i in range(repetations):
            input = residual_unit(nb_filter=nb_filter)(input)
        return input

    return f


def SE(input, ratio=8, channel=64):
    shared_layer_one = Dense(channel // ratio,
                             activation='relu')
    shared_layer_two = Dense(channel,
                             activation='tanh')
    avg_pool = GlobalAveragePooling2D(data_format='channels_first')(input)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    # 确定维度顺序
    cbam_feature = Permute((3, 1, 2))(avg_pool)
    # 逐元素相乘，得到下一步空间操作的输入feature
    return multiply([input, cbam_feature])


def stresnet_TaxiBJ_2D(c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32), t_conf=(3, 2, 32, 32), nb_residual_unit=5):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (len_seq, nb_flow, map_height, map_width)
    external_dim
    '''

    # main input
    global nb_flow, map_height, map_width
    main_inputs = []
    outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_flow, map_height, map_width = conf
            input = Input(shape=(nb_flow * len_seq, map_height, map_width))
            main_inputs.append(input)
            # Conv1
            conv1 = Convolution2D(
                nb_filter=64, nb_row=3, nb_col=3, border_mode="same", data_format='channels_first')(input)

            residual_output = ResUnits(_residual_unit, nb_filter=64,
                                       repetations=nb_residual_unit)(conv1)
            # Conv2
            activation = Activation('relu')(residual_output)
            conv2 = Convolution2D(
                nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same", data_format='channels_first')(activation)
            # outputs.append(conv2)

            # Multi-scale多尺度
            conv3_1 = Convolution2D(
                nb_filter=nb_flow, nb_row=1, nb_col=1, border_mode="same", data_format='channels_first')(conv2)
            conv3_2 = Convolution2D(
                nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same", data_format='channels_first')(conv2)
            conv3_3 = Convolution2D(
                nb_filter=nb_flow, nb_row=5, nb_col=5, border_mode="same", data_format='channels_first')(conv2)
            conv3 = Add()([conv3_1, conv3_2, conv3_3])
            outputs.append(conv3)

    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output))
        #        main_output = merge(new_outputs, mode='sum')
        main_output = Add()(new_outputs)
    main_output = Activation('relu', name='node_logits')(main_output)
    model = Model(input=main_inputs, output=main_output)

    return model
