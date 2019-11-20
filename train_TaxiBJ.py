from models.MyArima import *
import numpy as np
# from libs.utils import generate_x_y
from models.resnet_TaxiBj import stresnet_TaxiBJ
import keras.backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import config
import pandas as pd
import warnings
import os

# -*- coding:utf-8 -*-

import numpy as np


def search_day_data(train, num_of_days, label_start_idx, num_for_predict):
    '''
    find data in previous day given current start index.
    for example, if current start index is 8:00 am on Wed,
    it will return start and end index of 8:00 am on Tue

    Parameters
    ----------
    train: np.ndarray

    num_of_days: int, how many days will be used

    label_start_idx: current start index

    points_per_hour: number of points per hour

    num_for_predict: number of points will be predict

    Returns
    ----------
    list[(start_index, end_index)]: length is num_of_days, for example, if label_start_idx represents 8:00 am Wed,
                                    num_of_days is 2, it will return [(8:00 am Mon, 9:00 am Mon), (8:00 am Tue, 9:00 am Tue)]
    the second returned value is (label_start_idx, label_start_idx + num_for_predict), e.g. (8:00 am Wed, 9:00 am Wed)

    '''
    if label_start_idx + num_for_predict > len(train):
        return None
    x_idx = []
    for i in range(0, num_of_days):
        start_idx, end_idx = label_start_idx - 2 * 24 - i, label_start_idx - 2 * 24 - i + 1
        if start_idx >= 0 and end_idx >= 0:
            x_idx.append((start_idx, end_idx))
    if len(x_idx) != num_of_days:
        return None
    return list(reversed(x_idx)), (label_start_idx, label_start_idx + num_for_predict)


def search_day2_data(train, num_of_days, label_start_idx, num_for_predict):
    '''
    find data in previous day given current start index.
    for example, if current start index is 8:00 am on Wed,
    it will return start and end index of 8:00 am on Tue

    Parameters
    ----------
    train: np.ndarray

    num_of_days: int, how many days will be used

    label_start_idx: current start index

    points_per_hour: number of points per hour

    num_for_predict: number of points will be predict

    Returns
    ----------
    list[(start_index, end_index)]: length is num_of_days, for example, if label_start_idx represents 8:00 am Wed,
                                    num_of_days is 2, it will return [(8:00 am Mon, 9:00 am Mon), (8:00 am Tue, 9:00 am Tue)]
    the second returned value is (label_start_idx, label_start_idx + num_for_predict), e.g. (8:00 am Wed, 9:00 am Wed)

    '''
    if label_start_idx + num_for_predict > len(train):
        return None
    x_idx = []
    for i in range(0, num_of_days):
        start_idx, end_idx = label_start_idx - 2 * 2 * 24 - i, label_start_idx - 2 * 2 * 24 - i + 1
        if start_idx >= 0 and end_idx >= 0:
            x_idx.append((start_idx, end_idx))
    if len(x_idx) != num_of_days:
        return None
    return list(reversed(x_idx)), (label_start_idx, label_start_idx + num_for_predict)


def search_week_data(train, num_of_weeks, label_start_idx, num_for_predict):
    '''
    just like search_day_data, this function search previous week data
    '''

    # 最后一个预测样本往后就没了
    if label_start_idx + num_for_predict > len(train):
        return None
    x_idx = []
    # 封装第label_start_idx时间片对应的上一周的对应时间片的前5个时间片
    for i in range(0, num_of_weeks):
        start_idx, end_idx = label_start_idx - 2 * 24 * 5 - i, label_start_idx - 2 * 24 * 5 - i + 1
        if start_idx >= 0 and end_idx >= 0:
            x_idx.append((start_idx, end_idx))
    # 如果不够5个时间片，则该样本排除
    if len(x_idx) != num_of_weeks:
        return None
    # 返回上周的label_start_idx时间片对应的上一周的对应时间片的前5个时间片的序号
    return list(reversed(x_idx)), (label_start_idx, label_start_idx + num_for_predict)


def search_recent_data(train, num_of_hours, label_start_idx, num_for_predict):
    '''
    just like search_day_data, this function search previous hour data
    '''
    if label_start_idx + num_for_predict > len(train):
        return None
    x_idx = []
    for i in range(1, num_of_hours + 1):
        start_idx, end_idx = label_start_idx - i, label_start_idx - i + 1
        if start_idx >= 0 and end_idx >= 0:
            x_idx.append((start_idx, end_idx))
    if len(x_idx) != num_of_hours:
        return None
    return list(reversed(x_idx)), (label_start_idx, label_start_idx + num_for_predict)


def generate_x_y(train, num_of_weeks, num_of_days, num_of_hours, num_for_predict):
    '''
    Parameters
    ----------
    train: np.ndarray, shape is (num_of_samples, num_of_stations, num_of_flows)

    num_of_weeks=3, num_of_days=3, num_of_hours=3: int
    # 比如预测周二7：00-7：10的数据
    这里的num_of_weeks=3指的是上周周二{6:40-6:50,6:50-7:00,7:00-7:10},而不是指前三周
    同理num_of_days=3指的是前一天周一{6:40-6:50,6:50-7:00,7:00-7:10},而不是指前三天
    同理num_of_hours=3指的是今天周二{6：30-6：40,6:40-6:50,6:50-7:00}
    num_for_predict=1
    Returns
    ----------
    week_data: np.ndarray, shape is (num_of_samples, num_of_stations, num_of_flows, num_of_weeks)

    day_data: np.ndarray, shape is (num_of_samples, num_of_stations, num_of_flows, points_per_hour * num_of_days)

    recent_data: np.ndarray, shape is (num_of_samples, num_of_stations, num_of_flows, points_per_hour * num_of_hours)

    target: np.ndarray, shape is (num_of_samples, num_of_stations, num_for_predict)

    '''
    length = len(train)
    data = []
    # 根据recent\day\week来重新组织数据集
    for i in range(length):
        week = search_week_data(train, num_of_weeks, i, num_for_predict)
        day = search_day_data(train, num_of_days, i, num_for_predict)
        recent = search_recent_data(train, num_of_hours, i, num_for_predict)
        if week and day and recent:
            # 对应相同的预测值时才继续
            assert week[1] == day[1]
            assert day[1] == recent[1]

            week_data = np.concatenate([train[i: j] for i, j in week[0]], axis=0)
            day_data = np.concatenate([train[i: j] for i, j in day[0]], axis=0)
            recent_data = np.concatenate([train[i: j] for i, j in recent[0]], axis=0)
            data.append(((week_data, day_data, recent_data), train[week[1][0]: week[1][1]]))
    # ----通过上面的计算,data=(训练值(week, day, recent),真值)
    # zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    features, label = zip(*data)
    return features, label


def train_stresnet(Metro_Flow_Matrix):
    print(Metro_Flow_Matrix.shape)
    warnings.filterwarnings('ignore')

    N_days = 31  # 用了多少天的数据(目前17个工作日)
    N_hours = config.N_hours
    N_time_slice = 2  # 1小时有6个时间片
    N_station = 81  # 81个站点
    N_flow = config.N_flow  # 进站 & 出站
    len_seq1 = config.len_seq1  # week时间序列长度为2
    len_seq2 = config.len_seq2  # day时间序列长度为3
    len_seq3 = config.len_seq3  # hour时间序列长度为5
    nb_flow = config.nb_flow  # 输入特征

    # 自定义的损失函数
    # clip将超出指定范围的数强制变为边界值
    def my_own_loss_function(y_true, y_pred):
        return K.mean(abs(y_true - y_pred)) + 0.01 * K.mean(
            abs(K.clip(y_true, 0.001, 1) - K.clip(y_pred, 0.001, 1)) / K.clip(y_true, 0.001, 1)) * K.mean(
            abs(K.clip(y_true, 0.001, 1) - K.clip(y_pred, 0.001, 1)) / K.clip(y_true, 0.001, 1))

    # 学习率控制
    def scheduler(epoch):
        # 每隔15个epoch，学习率减小为原来的1/2
        if epoch % 15 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))
        return K.get_value(model.optimizer.lr)

    # ——————————————————————————————组织数据———————————————————————————————
    # 由于>25的数量极其少，将>25的值全都默认为25
    # Metro_Flow_Matrix[np.where(Metro_Flow_Matrix > 25)] = 25

    # 还是归一化
    # 点流/3000
    # 边流/233
    node_scale_ratio = 30  # 预处理第二步 相当于一共是/3000
    Metro_Flow_Matrix /= node_scale_ratio
    edge_scale_ratio = 233.0

    # 生成训练样本（也很关键）
    data, target = generate_x_y(Metro_Flow_Matrix, len_seq1, len_seq2, len_seq3, 1)  # type为tuple

    # tuple to array
    # zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    week, day, hour = zip(*data)
    # 矩阵转换为array
    node_data_1 = np.array(week)
    node_data_2 = np.array(day)
    node_data_3 = np.array(hour)
    # print("node_data_1.shape=", node_data_1.shape, "node_data_2.shape=", node_data_2.shape, "node_data_3.shape=",
    #       node_data_3.shape)

    # print("edge_data_1.shape=", edge_data_1.shape, "edge_data_2.shape=", edge_data_2.shape, "edge_data_3.shape=",
    #       edge_data_3.shape)

    target = np.array(target)
    # print("target.shape=", target.shape)

    # ——————————————————————————————重新组织数据——————————————————————————————————
    # 将data切割出recent\period\trend数据
    length = node_data_1.shape[0]
    xr_train = np.zeros([length, N_station, len_seq3 * N_flow])
    xp_train = np.zeros([length, N_station, len_seq2 * nb_flow])
    xt_train = np.zeros([length, N_station, len_seq1 * nb_flow])
    xredge_train = np.zeros([length, N_station, len_seq3 * N_station])
    xpedge_train = np.zeros([length, N_station, len_seq2 * N_station])
    xtedge_train = np.zeros([length, N_station, len_seq1 * N_station])
    # 装载xr_train, xp_train, xt_train等最终样本，所以len_seq * nb_flow = 3*2
    for i in range(length):
        for j in range(len_seq3):
            for k in range(2):
                # 组装当前前5个时间片矩阵
                xr_train[i, :, j * 2 + k] = node_data_3[i, j, :, k]
    for i in range(length):
        for j in range(len_seq2):
            for k in range(2):
                # 组装前一天前3个时间片矩阵
                xp_train[i, :, j * 2 + k] = node_data_2[i, j, :, k]
    for i in range(length):
        for j in range(len_seq1):
            for k in range(2):
                # 组装前一周对应的前2个时间片矩阵
                xt_train[i, :, j * 2 + k] = node_data_1[i, j, :, k]

    # ——————————————————————————————SHUFFLE—————————————————————————————————————————————
    indices = np.arange(length)
    # 打乱
    np.random.shuffle(indices)
    xr_train = xr_train[indices]  # learning task: (2302, 81, 10)->(2302, 81, 2)
    xp_train = xp_train[indices]
    xt_train = xt_train[indices]
    # 1、2、4、9为Embedding, 5为Dense
    # print("x_train_external_information1.shape=", x_train_external_information1.shape)
    target = target[indices]
    # ————————————————————————————————构建验证集合(24-25号数据作为验证集)—————————————————————————————————————
    # 构造得有点复杂.....2019.05.22
    node_day_24 = np.load('./npy/test_data/taxibj_node_data_day0403.npy')[:, 0:81, :]
    node_day_25 = np.load('./npy/test_data/taxibj_node_data_day0404.npy')[:, 0:81, :]
    node_day_18 = np.load('./npy/test_data/taxibj_node_data_day0329.npy')[:, 0:81, :]
    node_day_18 = np.vstack((node_day_18, node_day_18))
    node_day_18 = np.vstack((node_day_18, node_day_18))
    val_node_data = np.concatenate((node_day_18, node_day_24, node_day_25), axis=0)
    print(val_node_data.shape)

    # raw_edge_data = np.load('./npy/train_data/raw_edge_data.npy')
    # edge_day_24 = raw_edge_data[-144:, :, :]
    # edge_day_18 = raw_edge_data[144 * 12:-144, :, :]
    # edge_day_25 = np.load('./npy/train_data/day_25_edge.npy')
    # val_edge_data = np.concatenate((edge_day_18, edge_day_24, edge_day_25), axis=0)

    # normalization
    val_node_data /= node_scale_ratio
    # val_edge_data /= edge_scale_ratio

    # 构建好验证集的样本和标签
    val_node_data, val_node_target = generate_x_y(val_node_data, len_seq1, len_seq2, len_seq3, 1)
    # val_edge_data, val_edge_target = generate_x_y(val_edge_data, len_seq1, len_seq2, len_seq3, 1)

    # tuple-->array
    week, day, hour = zip(*val_node_data)
    val_node_data_1 = np.array(week)
    val_node_data_2 = np.array(day)
    val_node_data_3 = np.array(hour)

    # week, day, hour = zip(*val_edge_data)
    # val_edge_data_1 = np.array(week)
    # val_edge_data_2 = np.array(day)
    # val_edge_data_3 = np.array(hour)

    val_node_target = np.array(val_node_target)
    # val_edge_target = np.array(val_edge_target)

    # # 将data切割出recent\period\trend数据
    val_length = val_node_data_1.shape[0]
    xr_val = np.zeros([val_length, N_station, len_seq3 * N_flow])
    xp_val = np.zeros([val_length, N_station, len_seq2 * N_flow])
    xt_val = np.zeros([val_length, N_station, len_seq1 * N_flow])
    # xredge_val = np.zeros([val_length, N_station, len_seq3 * N_station])
    # xpedge_val = np.zeros([val_length, N_station, len_seq2 * N_station])
    # xtedge_val = np.zeros([val_length, N_station, len_seq1 * N_station])
    # # 适应st_resnet，由于没有LSTM，所以len_seq * nb_flow = 3*2
    for i in range(val_length):
        for j in range(len_seq3):
            for k in range(2):
                xr_val[i, :, j * 2 + k] = val_node_data_3[i, j, :, k]
    for i in range(val_length):
        for j in range(len_seq2):
            for k in range(2):
                # print(val_node_data_2[i, j, :, k])
                xp_val[i, :, j * 2 + k] = val_node_data_2[i, j, :, k]
    for i in range(val_length):
        for j in range(len_seq1):
            for k in range(2):
                xt_val[i, :, j * 2 + k] = val_node_data_1[i, j, :, k]
    #
    # for i in range(val_length):
    #     for j in range(len_seq3):
    #         for k in range(81):
    #             xredge_val[i, :, j * 81 + k] = val_edge_data_3[i, j, :, k]
    # for i in range(val_length):
    #     for j in range(len_seq2):
    #         for k in range(81):
    #             xpedge_val[i, :, j * 81 + k] = val_edge_data_2[i, j, :, k]
    # for i in range(val_length):
    #     for j in range(len_seq1):
    #         for k in range(81):
    #             xtedge_val[i, :, j * 81 + k] = val_edge_data_1[i, j, :, k]
    # # ——————————————————————添加验证集外部信息————————————————————————
    # # 默认了00:00-00:40的人流量均为0
    # # 0125是周五,weekday信息
    # x_val_external_information1 = np.zeros([24 * 6, 1])
    # x_val_external_information1[:, 0] = 4  # 代表周五
    # # 时间片信息
    # x_val_external_information2 = np.zeros([24 * 6, 1])
    # HOUR = 0
    # for i in range(0, 24 * 6):
    #     x_val_external_information2[i, 0] = HOUR
    #     HOUR = HOUR + 1
    # # 天气信息,25号为多云
    # x_val_external_information4 = np.zeros([24 * 6, 1])
    # x_val_external_information4[:, 0] = 1
    # # 闸机信息
    # x_val_external_information5 = np.zeros([24 * 6, 81])
    # t = np.load('./npy/train_data/Two_more_features.npy')
    # t = t[:, 0]
    # for i in range(144):
    #     x_val_external_information5[i, :] = t
    # # 早晚高峰、高峰、平峰信息
    # x_val_external_information9 = np.zeros([N_hours * N_time_slice, 1])
    # # #——————————————————早晚高峰—————————————————————
    # x_val_external_information9[39:54, 0] = 2  # 7：30 - 9：00
    # x_val_external_information9[102:114, 0] = 2  # 17：00 - 19：00
    # # #——————————————————高峰—————————————————————————
    # x_val_external_information9[33:39, 0] = 1  # 6:30-7:30
    # x_val_external_information9[63:70, 0] = 1  # 10:30-11:30
    # x_val_external_information9[99:102, 0] = 1  # 16:30-17:30
    # x_val_external_information9[114:132, 0] = 1  # 19:00-22:00
    # #------------val external information往后移一个时间片------------------
    # x_val_external_information1 = x_val_external_information1[len(x_val_external_information1) - len(xr_val):, :]
    # x_val_external_information2 = x_val_external_information2[len(x_val_external_information2) - len(xr_val):, :]
    # x_val_external_information4 = x_val_external_information4[len(x_val_external_information4) - len(xr_val):, :]
    # x_val_external_information5 = x_val_external_information5[len(x_val_external_information5) - len(xr_val):, :]
    # x_val_external_information9 = x_val_external_information9[len(x_val_external_information9) - len(xr_val):, :]

    # 重新reshape一下
    val_node_target = val_node_target.reshape(
        [val_node_target.shape[0], val_node_target.shape[1] * val_node_target.shape[2], val_node_target.shape[3]])
    target = target.reshape([target.shape[0], target.shape[1] * target.shape[2], target.shape[3]])

    # 这里开始跑模型，神经网络相关的，重点看这里
    # ——————————————————————————————建立模型—————————————————————————————————————
    model = stresnet_TaxiBJ(c_conf=(len_seq3, N_flow, N_station), p_conf=(len_seq2, N_flow, N_station),
                            t_conf=(len_seq1, N_flow, N_station), nb_residual_unit=4)  # 这里的unit代表了大体的网络深度
    model.compile(optimizer='adam',
                  loss={'node_logits': my_own_loss_function}, metrics=['accuracy'])

    filepath = "./log/stresnet/TaxiBJ/" + "{epoch:02d}-{loss:.8f}.hdf5"
    if not os.path.exists("./log/stresnet/TaxiBJ/"):
        os.makedirs("./log/stresnet/TaxiBJ/")
    # 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True,
                                 mode='min')
    reduce_lr = LearningRateScheduler(scheduler)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='min')
    callbacks_list = [checkpoint, reduce_lr, early_stopping]
    K.set_value(model.optimizer.lr, 0.0001)
    # 输入的东西要注意

    history = model.fit([xr_train, xp_train, xt_train],
                        [target],
                        validation_data=([xr_val, xp_val, xt_val],
                                         [val_node_target]),
                        batch_size=config.batch_size, epochs=config.epochs, callbacks=callbacks_list)

    pd.DataFrame(columns=['loss'], data=history.history['loss']).to_csv(config.loss_acc_csvFile, index=None)


if __name__ == '__main__':
    # --------------------------------stresnet训练-start---------------------------------------#

    # 直接从这里开始看，程序的入口，加载统计好的流量文件
    Metro_Flow_Matrix = np.load('./npy/train_data/taxibj_node_data_month3_23.npy')  # shape=(2448, 81, 2)
    Metro_Flow_Matrix = Metro_Flow_Matrix.reshape(
        [Metro_Flow_Matrix.shape[0], Metro_Flow_Matrix.shape[2] * Metro_Flow_Matrix.shape[3],
         Metro_Flow_Matrix.shape[1]])
    Metro_Flow_Matrix = Metro_Flow_Matrix[:, 0:81, :]
    train_stresnet(Metro_Flow_Matrix)

    # --------------------------------stresnet训练-end---------------------------------------#