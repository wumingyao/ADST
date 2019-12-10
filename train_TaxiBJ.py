from models.MyArima import *
from models.STResNet_TaxiBJ import stresnet_TaxiBJ_2D
from models.resnet_TaxiBj import stresnet_TaxiBJ
from models.LSTM_TaxiBJ import lstm_TaxiBJ
import keras.backend as K
from keras.optimizers import Adam
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

    # 还是归一化
    # 点流/3000
    # 边流/233
    node_scale_ratio = 30  # 预处理第二步 相当于一共是/3000
    Metro_Flow_Matrix /= node_scale_ratio

    # 生成训练样本（也很关键）
    data, target = generate_x_y(Metro_Flow_Matrix, len_seq1, len_seq2, len_seq3, 1)  # type为tuple

    # tuple to array
    # zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    week, day, hour = zip(*data)
    # 矩阵转换为array
    node_data_1 = np.array(week)
    node_data_2 = np.array(day)
    node_data_3 = np.array(hour)

    target = np.array(target)

    # ——————————————————————————————重新组织数据——————————————————————————————————
    # 将data切割出recent\period\trend数据
    length = node_data_1.shape[0]
    xr_train = np.zeros([length, N_station, len_seq3 * N_flow])
    xp_train = np.zeros([length, N_station, len_seq2 * nb_flow])
    xt_train = np.zeros([length, N_station, len_seq1 * nb_flow])
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
    target = target[indices]
    # ————————————————————————————————构建验证集合(24-25号数据作为验证集)—————————————————————————————————————
    # 构造得有点复杂.....2019.05.22
    node_day_24 = np.load('./npy/test_data/taxibj_node_data_day0403.npy')[:, 0:81, :]
    node_day_25 = np.load('./npy/test_data/taxibj_node_data_day0404.npy')[:, 0:81, :]
    node_day_18 = np.load('./npy/test_data/taxibj_node_data_day0329.npy')[:, 0:81, :]
    node_day_18 = np.vstack((node_day_18, node_day_18))
    node_day_18 = np.vstack((node_day_18, node_day_18))
    val_node_data = np.concatenate((node_day_18, node_day_24, node_day_25), axis=0)

    # normalization
    val_node_data /= node_scale_ratio

    # 构建好验证集的样本和标签
    val_node_data, val_node_target = generate_x_y(val_node_data, len_seq1, len_seq2, len_seq3, 1)

    # tuple-->array
    week, day, hour = zip(*val_node_data)
    val_node_data_1 = np.array(week)
    val_node_data_2 = np.array(day)
    val_node_data_3 = np.array(hour)

    val_node_target = np.array(val_node_target)

    # # 将data切割出recent\period\trend数据
    val_length = val_node_data_1.shape[0]
    xr_val = np.zeros([val_length, N_station, len_seq3 * N_flow])
    xp_val = np.zeros([val_length, N_station, len_seq2 * N_flow])
    xt_val = np.zeros([val_length, N_station, len_seq1 * N_flow])
    # # 适应st_resnet，由于没有LSTM，所以len_seq * nb_flow = 3*2
    for i in range(val_length):
        for j in range(len_seq3):
            for k in range(2):
                xr_val[i, :, j * 2 + k] = val_node_data_3[i, j, :, k]
    for i in range(val_length):
        for j in range(len_seq2):
            for k in range(2):
                xp_val[i, :, j * 2 + k] = val_node_data_2[i, j, :, k]
    for i in range(val_length):
        for j in range(len_seq1):
            for k in range(2):
                xt_val[i, :, j * 2 + k] = val_node_data_1[i, j, :, k]

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


def train_LSTM(Metro_Flow_Matrix):
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

    # 还是归一化
    # 点流/3000
    # 边流/233
    node_scale_ratio = 30  # 预处理第二步 相当于一共是/3000
    Metro_Flow_Matrix /= node_scale_ratio

    # 生成训练样本（也很关键）
    data, target = generate_x_y(Metro_Flow_Matrix, len_seq1, len_seq2, len_seq3, 1)  # type为tuple

    # tuple to array
    # zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    week, day, hour = zip(*data)
    # 矩阵转换为array
    node_data_1 = np.array(week)
    node_data_2 = np.array(day)
    node_data_3 = np.array(hour)

    target = np.array(target)

    # ——————————————————————————————重新组织数据——————————————————————————————————
    # 将data切割出recent\period\trend数据
    length = node_data_1.shape[0]
    xr_train = np.zeros([length, N_station, len_seq3 * N_flow])
    xp_train = np.zeros([length, N_station, len_seq2 * nb_flow])
    xt_train = np.zeros([length, N_station, len_seq1 * nb_flow])
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
    target = target[indices]
    # ————————————————————————————————构建验证集合(24-25号数据作为验证集)—————————————————————————————————————
    # 构造得有点复杂.....2019.05.22
    node_day_24 = np.load('./npy/test_data/taxibj_node_data_day0403.npy')[:, 0:81, :]
    node_day_25 = np.load('./npy/test_data/taxibj_node_data_day0404.npy')[:, 0:81, :]
    node_day_18 = np.load('./npy/test_data/taxibj_node_data_day0329.npy')[:, 0:81, :]
    node_day_18 = np.vstack((node_day_18, node_day_18))
    node_day_18 = np.vstack((node_day_18, node_day_18))
    val_node_data = np.concatenate((node_day_18, node_day_24, node_day_25), axis=0)

    # normalization
    val_node_data /= node_scale_ratio

    # 构建好验证集的样本和标签
    val_node_data, val_node_target = generate_x_y(val_node_data, len_seq1, len_seq2, len_seq3, 1)

    # tuple-->array
    week, day, hour = zip(*val_node_data)
    val_node_data_1 = np.array(week)
    val_node_data_2 = np.array(day)
    val_node_data_3 = np.array(hour)

    val_node_target = np.array(val_node_target)

    # # 将data切割出recent\period\trend数据
    val_length = val_node_data_1.shape[0]
    xr_val = np.zeros([val_length, N_station, len_seq3 * N_flow])
    xp_val = np.zeros([val_length, N_station, len_seq2 * N_flow])
    xt_val = np.zeros([val_length, N_station, len_seq1 * N_flow])
    # # 适应st_resnet，由于没有LSTM，所以len_seq * nb_flow = 3*2
    for i in range(val_length):
        for j in range(len_seq3):
            for k in range(2):
                xr_val[i, :, j * 2 + k] = val_node_data_3[i, j, :, k]
    for i in range(val_length):
        for j in range(len_seq2):
            for k in range(2):
                xp_val[i, :, j * 2 + k] = val_node_data_2[i, j, :, k]
    for i in range(val_length):
        for j in range(len_seq1):
            for k in range(2):
                xt_val[i, :, j * 2 + k] = val_node_data_1[i, j, :, k]

    # 重新reshape一下
    val_node_target = val_node_target.reshape(
        [val_node_target.shape[0], val_node_target.shape[1] * val_node_target.shape[2], val_node_target.shape[3]])
    target = target.reshape([target.shape[0], target.shape[1] * target.shape[2], target.shape[3]])

    # 这里开始跑模型，神经网络相关的，重点看这里
    # ——————————————————————————————建立模型—————————————————————————————————————
    model = lstm_TaxiBJ(c_conf=(len_seq3, N_flow, N_station), p_conf=(len_seq2, N_flow, N_station),
                        t_conf=(len_seq1, N_flow, N_station))  # 这里的unit代表了大体的网络深度
    model.compile(optimizer='adam',
                  loss={'node_logits': my_own_loss_function}, metrics=['accuracy'])

    filepath = "./log/LSTM/TaxiBJ/" + "{epoch:02d}-{loss:.8f}.hdf5"
    if not os.path.exists("./log/LSTM/TaxiBJ/"):
        os.makedirs("./log/LSTM/TaxiBJ/")
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


def train_Arima_TaxiBJ(data):
    # 读数据
    N_time_slice = 2
    predict_arima = np.zeros(shape=(config.N_hours * N_time_slice, config.N_station, config.N_flow))
    data = data / 30
    arima_para = {}
    arima_para['p'] = 1
    arima_para['d'] = 0
    arima_para['q'] = 0
    arima = Arima_Class(arima_para)

    for i in range(config.N_station - 1):
        # 对第i个站点
        for j in range(config.N_flow - 1):
            try:
                # 对于出站或入站
                tr_data = data[:, i:i + 1, j:j + 1]
                tr_data = np.squeeze(tr_data)
                model = arima.fit(tr_data)
                predict = arima.pred(model, 48)
                predict = predict.reshape(len(predict), 1, 1)
                predict_arima[:, i:i + 1, j:j + 1] = predict
            except Exception:
                continue
    # 后处理
    predict_arima = np.where(predict_arima > 0, predict_arima, 0)
    return predict_arima


def train_stresnet_TaxiBJ_2D(Metro_Flow_Matrix):
    warnings.filterwarnings('ignore')
    # conf = (len_seq, nb_flow, map_height, map_width)
    N_days = 31  # 用了多少天的数据(目前17个工作日)
    N_hours = config.N_hours
    N_time_slice = 2  # 1小时有6个时间片
    N_station = 81  # 81个站点
    map_height = 32
    map_width = 32
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

    # 还是归一化
    # 点流/3000
    # 边流/233
    node_scale_ratio = 30  # 预处理第二步 相当于一共是/3000
    Metro_Flow_Matrix /= node_scale_ratio

    # 生成训练样本（也很关键）
    data, target = generate_x_y(Metro_Flow_Matrix, len_seq1, len_seq2, len_seq3, 1)  # type为tuple

    # tuple to array
    # zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    week, day, hour = zip(*data)
    # 矩阵转换为array
    node_data_1 = np.array(week)
    node_data_2 = np.array(day)
    node_data_3 = np.array(hour)

    target = np.array(target)

    # ——————————————————————————————重新组织数据——————————————————————————————————
    # 将data切割出recent\period\trend数据
    length = node_data_1.shape[0]
    # len_seq, nb_flow, map_height, map_width
    xr_train = np.zeros([length, len_seq3, N_flow, map_height, map_width])
    xp_train = np.zeros([length, len_seq2, N_flow, map_height, map_width])
    xt_train = np.zeros([length, len_seq1, N_flow, map_height, map_width])
    # 装载xr_train, xp_train, xt_train等最终样本，所以len_seq * nb_flow = 3*2
    for i in range(length):
        for j in range(len_seq3):
            for k in range(2):
                # 组装当前前5个时间片矩阵
                xr_train[i, j, k, :, :] = node_data_3[i, j, k, :, :]
    for i in range(length):
        for j in range(len_seq2):
            for k in range(2):
                # 组装前一天前3个时间片矩阵
                xp_train[i, j, k, :, :] = node_data_2[i, j, k, :, :]
    for i in range(length):
        for j in range(len_seq1):
            for k in range(2):
                # 组装前一周对应的前2个时间片矩阵
                xt_train[i, j, k, :, :] = node_data_1[i, j, k, :, :]
    # ——————————————————————————————SHUFFLE—————————————————————————————————————————————
    indices = np.arange(length)
    # 打乱
    np.random.shuffle(indices)
    xr_train = xr_train[indices]  # learning task: (2302, 81, 10)->(2302, 81, 2)
    xp_train = xp_train[indices]
    xt_train = xt_train[indices]
    # 1、2、4、9为Embedding, 5为Dense
    target = target[indices]
    # ————————————————————————————————构建验证集合(24-25号数据作为验证集)—————————————————————————————————————
    # 构造得有点复杂.....2019.05.22
    node_day_24 = np.load('./npy/test_data/taxibj_node_data_day0403.npy')
    node_day_24 = node_day_24.reshape([node_day_24.shape[0], node_day_24.shape[2], 32, 32])

    node_day_25 = np.load('./npy/test_data/taxibj_node_data_day0404.npy')
    node_day_25 = node_day_25.reshape([node_day_25.shape[0], node_day_25.shape[2], 32, 32])
    node_day_18 = np.load('./npy/test_data/taxibj_node_data_day0329.npy')
    node_day_18 = node_day_18.reshape([node_day_18.shape[0], node_day_18.shape[2], 32, 32])
    node_day_18 = np.vstack((node_day_18, node_day_18))
    node_day_18 = np.vstack((node_day_18, node_day_18))
    val_node_data = np.concatenate((node_day_18, node_day_24, node_day_25), axis=0)

    # normalization
    val_node_data /= node_scale_ratio

    # 构建好验证集的样本和标签
    val_node_data, val_node_target = generate_x_y(val_node_data, len_seq1, len_seq2, len_seq3, 1)

    # tuple-->array
    week, day, hour = zip(*val_node_data)
    val_node_data_1 = np.array(week)
    val_node_data_2 = np.array(day)
    val_node_data_3 = np.array(hour)

    val_node_target = np.array(val_node_target)

    # # 将data切割出recent\period\trend数据
    val_length = val_node_data_1.shape[0]
    xr_val = np.zeros([val_length, len_seq3, N_flow, map_height, map_width])
    xp_val = np.zeros([val_length, len_seq2, N_flow, map_height, map_width])
    xt_val = np.zeros([val_length, len_seq1, N_flow, map_height, map_width])
    # # 适应st_resnet，由于没有LSTM，所以len_seq * nb_flow = 3*2
    for i in range(val_length):
        for j in range(len_seq3):
            for k in range(2):
                xr_val[i, j, k, :, :] = val_node_data_3[i, j, k, :, :]
    for i in range(val_length):
        for j in range(len_seq2):
            for k in range(2):
                xp_val[i, j, k, :, :] = val_node_data_2[i, j, k, :, :]
    for i in range(val_length):
        for j in range(len_seq1):
            for k in range(2):
                xt_val[i, j, k, :, :] = val_node_data_1[i, j, k, :, :]

    # 重新reshape一下，以适应网络
    xr_train = xr_train.reshape(
        [xr_train.shape[0], xr_train.shape[1] * xr_train.shape[2], xr_train.shape[3], xr_train.shape[4]])
    xp_train = xp_train.reshape(
        [xp_train.shape[0], xp_train.shape[1] * xp_train.shape[2], xp_train.shape[3], xp_train.shape[4]])
    xt_train = xt_train.reshape(
        [xt_train.shape[0], xt_train.shape[1] * xt_train.shape[2], xt_train.shape[3], xt_train.shape[4]])
    target = np.squeeze(target, axis=1)

    xr_val = xr_val.reshape(
        [xr_val.shape[0], xr_val.shape[1] * xr_val.shape[2], xr_val.shape[3], xr_val.shape[4]])
    xp_val = xp_val.reshape(
        [xp_val.shape[0], xp_val.shape[1] * xp_val.shape[2], xp_val.shape[3], xp_val.shape[4]])
    xt_val = xt_val.reshape(
        [xt_val.shape[0], xt_val.shape[1] * xt_val.shape[2], xt_val.shape[3], xt_val.shape[4]])
    val_node_target = np.squeeze(val_node_target, axis=1)

    # 这里开始跑模型，神经网络相关的，重点看这里
    # ——————————————————————————————建立模型—————————————————————————————————————
    model = stresnet_TaxiBJ_2D(c_conf=(len_seq3, nb_flow, map_height, map_width),
                               p_conf=(len_seq2, nb_flow, map_height, map_width),
                               t_conf=(len_seq1, nb_flow, map_height, map_width),
                               nb_residual_unit=4)  # 这里的unit代表了大体的网络深度
    # model.compile(optimizer='adam',
    #               loss={'node_logits': my_own_loss_function}, metrics=['accuracy'])
    model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

    filepath = "./log/stresnet/TaxiBJ_2D/" + "{epoch:02d}-{loss:.8f}.hdf5"
    if not os.path.exists("./log/stresnet/TaxiBJ_2D/"):
        os.makedirs("./log/stresnet/TaxiBJ_2D/")
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
    # # --------------------------------stresnet训练-start---------------------------------------#
    #
    # # 直接从这里开始看，程序的入口，加载统计好的流量文件
    # Metro_Flow_Matrix = np.load('./npy/train_data/taxibj_node_data_month3_23.npy')  # shape=(2448, 81, 2)
    # Metro_Flow_Matrix = Metro_Flow_Matrix.reshape(
    #     [Metro_Flow_Matrix.shape[0], Metro_Flow_Matrix.shape[2] * Metro_Flow_Matrix.shape[3],
    #      Metro_Flow_Matrix.shape[1]])
    # Metro_Flow_Matrix = Metro_Flow_Matrix[:, 0:81, :]
    # train_stresnet(Metro_Flow_Matrix)
    #
    # # --------------------------------stresnet训练-end---------------------------------------#
    #
    # # --------------------------------LSTM训练-start---------------------------------------#
    #
    # # 直接从这里开始看，程序的入口，加载统计好的流量文件
    # Metro_Flow_Matrix = np.load('./npy/train_data/taxibj_node_data_month3_23.npy')  # shape=(2448, 81, 2)
    # Metro_Flow_Matrix = Metro_Flow_Matrix.reshape(
    #     [Metro_Flow_Matrix.shape[0], Metro_Flow_Matrix.shape[2] * Metro_Flow_Matrix.shape[3],
    #      Metro_Flow_Matrix.shape[1]])
    # Metro_Flow_Matrix = Metro_Flow_Matrix[:, 0:81, :]
    # train_LSTM(Metro_Flow_Matrix)
    #
    # # --------------------------------LSTM训练-end---------------------------------------#
    #
    # # --------------------------------Arima训练_0402-start---------------------------------------#
    # data_train = np.load('./npy/train_data/taxibj_node_data_month3_23.npy')
    # data_train = data_train.reshape(
    #     [data_train.shape[0], data_train.shape[2] * data_train.shape[3],
    #      data_train.shape[1]])
    # data_train = data_train[:, 0:81, :]
    # predict_arima = train_Arima_TaxiBJ(data_train)
    # np.save('./npy/mae_compare/predict_arima_TaxiBj_day0402.npy', predict_arima)
    # # --------------------------------Arima训练-end---------------------------------------#
    # # --------------------------------Arima训练_0403-start---------------------------------------#
    # data_train = np.load('./npy/train_data/taxibj_node_data_month3_23.npy')
    # data_train1 = np.load('./npy/test_data/taxibj_node_data_day0402.npy')
    # data_train = data_train.reshape(
    #     [data_train.shape[0], data_train.shape[2] * data_train.shape[3],
    #      data_train.shape[1]])
    # data_train = np.vstack((data_train, data_train1))
    # data_train = data_train[:, 0:81, :]
    # predict_arima = train_Arima_TaxiBJ(data_train)
    # np.save('./npy/mae_compare/predict_arima_TaxiBj_day0403.npy', predict_arima)
    # # --------------------------------Arima训练_0403-end---------------------------------------#
    #
    # # --------------------------------Arima训练_0404-start---------------------------------------#
    # data_train = np.load('./npy/train_data/taxibj_node_data_month3_23.npy')
    # data_train1 = np.load('./npy/test_data/taxibj_node_data_day0402.npy')
    # data_train2 = np.load('./npy/test_data/taxibj_node_data_day0403.npy')
    # data_train = data_train.reshape(
    #     [data_train.shape[0], data_train.shape[2] * data_train.shape[3],
    #      data_train.shape[1]])
    # data_train = np.vstack((data_train, data_train1, data_train2))
    # data_train = data_train[:, 0:81, :]
    # predict_arima = train_Arima_TaxiBJ(data_train)
    # np.save('./npy/mae_compare/predict_arima_TaxiBj_day0404.npy', predict_arima)
    # # --------------------------------Arima训练_0404-end---------------------------------------#
    #
    # # --------------------------------Arima训练_0405-start---------------------------------------#
    # data_train = np.load('./npy/train_data/taxibj_node_data_month3_23.npy')
    # data_train1 = np.load('./npy/test_data/taxibj_node_data_day0402.npy')
    # data_train2 = np.load('./npy/test_data/taxibj_node_data_day0403.npy')
    # data_train3 = np.load('./npy/test_data/taxibj_node_data_day0404.npy')
    # data_train = data_train.reshape(
    #     [data_train.shape[0], data_train.shape[2] * data_train.shape[3],
    #      data_train.shape[1]])
    # data_train = np.vstack((data_train, data_train1, data_train2, data_train3))
    # data_train = data_train[:, 0:81, :]
    # predict_arima = train_Arima_TaxiBJ(data_train)
    # np.save('./npy/mae_compare/predict_arima_TaxiBj_day0405.npy', predict_arima)
    # # --------------------------------Arima训练_0405-end---------------------------------------#
    # --------------------------------stresnet_2D训练-start---------------------------------------#

    # 直接从这里开始看，程序的入口，加载统计好的流量文件
    Metro_Flow_Matrix = np.load('./npy/train_data/taxibj_node_data_month3_23.npy')
    # Metro_Flow_Matrix = Metro_Flow_Matrix.reshape(
    #     [Metro_Flow_Matrix.shape[0], Metro_Flow_Matrix.shape[2] * Metro_Flow_Matrix.shape[3],
    #      Metro_Flow_Matrix.shape[1]])
    # Metro_Flow_Matrix = Metro_Flow_Matrix[:, 0:81, :]
    train_stresnet_TaxiBJ_2D(Metro_Flow_Matrix)

    # --------------------------------stresnet训练-end---------------------------------------#
