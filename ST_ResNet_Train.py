import numpy as np
from libs.utils import generate_x_y
from models.STResNet import stresnet
import keras.backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import config
import pandas as pd
import warnings


def train_stresnet(Metro_Flow_Matrix, Metro_Edge_Flow_Matrix):
    warnings.filterwarnings('ignore')

    N_days = config.N_days  # 用了多少天的数据(目前17个工作日)
    N_hours = config.N_hours
    N_time_slice = config.N_time_slice  # 1小时有6个时间片
    N_station = config.N_station  # 81个站点
    N_flow = config.N_flow  # 进站 & 出站
    len_seq1 = config.len_seq1  # week时间序列长度为2
    len_seq2 = config.len_seq2  # day时间序列长度为3
    len_seq3 = config.len_seq3  # hour时间序列长度为5
    len_pre = config.len_pre
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
    Metro_Edge_Flow_Matrix /= edge_scale_ratio

    # 生成训练样本（也很关键）
    data, target = generate_x_y(Metro_Flow_Matrix, len_seq1, len_seq2, len_seq3, len_pre)  # type为tuple
    edge_data, edge_target = generate_x_y(Metro_Edge_Flow_Matrix, len_seq1, len_seq2, len_seq3, len_pre)

    # tuple to array
    # zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    week, day, hour = zip(*data)
    # 矩阵转换为array
    node_data_1 = np.array(week)
    node_data_2 = np.array(day)
    node_data_3 = np.array(hour)

    week, day, hour = zip(*edge_data)
    edge_data_1 = np.array(week)
    edge_data_2 = np.array(day)
    edge_data_3 = np.array(hour)

    target = np.array(target)
    edge_target = np.array(edge_target)

    # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    target = np.squeeze(target, axis=1)
    edge_target = np.squeeze(edge_target, axis=1)
    # 根据edge_target获取mask矩阵
    mask = edge_target == 0
    mx = np.ma.array(edge_target, mask=mask)
    mask_matrix = mx.mask + 0
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

    for i in range(length):
        for j in range(len_seq1):
            for k in range(81):
                xtedge_train[i, :, j * 81 + k] = edge_data_1[i, j, :, k]
    for i in range(length):
        for j in range(len_seq2):
            for k in range(81):
                xpedge_train[i, :, j * 81 + k] = edge_data_2[i, j, :, k]
    for i in range(length):
        for j in range(len_seq3):
            for k in range(81):
                xredge_train[i, :, j * 81 + k] = edge_data_3[i, j, :, k]

    # # ——————————————————————————————加入外部信息-周信息——————————————————————————————————
    #
    # DAY = 3  # 训练样本从3号开始，为周四，最后需要移动5个时间片[0,1,2,3,4]
    # x_external_information1 = np.zeros([(N_days - 1) * N_hours * N_time_slice, 1])
    # # range(start, stop[, step])
    # # 0到16
    # for i in range(0, (N_days - 1) * N_hours * N_time_slice, 24 * 6):
    #     # 标记每个样本属于哪一天
    #     x_external_information1[i:i + 24 * 6, 0] = DAY
    #     # 这里为什么是5
    #     DAY = (DAY + 1) % 5
    #
    # # ——————————————————————————————加入外部信息-小时&分钟信息——————————————————————————————————
    # HOUR = 0  # 从0时刻开始，最后需要移动5个时间片[0,1,2,...,24*6-1]
    # x_external_information2 = np.zeros([(N_days - 1) * N_hours * N_time_slice, 1])
    # for i in range(0, (N_days - 1) * N_hours * N_time_slice):
    #     x_external_information2[i, 0] = HOUR
    #     HOUR = (HOUR + 1) % (24 * 6)
    #
    # # ——————————————————————————————加入外部信息-天气信息—————————————————————————————————
    # x_external_information4 = np.zeros([N_days * N_hours * N_time_slice, 1])
    # # [中雨、小雨、阴、多云、晴] --> 简化情况为[雨天/晴天] 0/1
    # # 2号--阴
    # x_external_information4[0:24 * 6, 0] = 1
    # # 3号--小雨/阴
    # x_external_information4[24 * 6:2 * 24 * 6, 0] = 0
    # # 4号--中雨/小雨
    # x_external_information4[2 * 24 * 6:3 * 24 * 6, 0] = 0
    # # 7号--小雨
    # x_external_information4[3 * 24 * 6:4 * 24 * 6, 0] = 0
    # # 8号--小雨/阴
    # x_external_information4[4 * 24 * 6:5 * 24 * 6, 0] = 0
    # # 9号--中雨/小雨
    # x_external_information4[5 * 24 * 6:6 * 24 * 6, 0] = 0
    # # 10号--小雨
    # x_external_information4[6 * 24 * 6:7 * 24 * 6, 0] = 0
    # # 11号--小雨
    # x_external_information4[7 * 24 * 6:8 * 24 * 6, 0] = 0
    # # 14号--小雨
    # x_external_information4[8 * 24 * 6:9 * 24 * 6, 0] = 0
    # # 15号--小雨
    # x_external_information4[9 * 24 * 6:10 * 24 * 6, 0] = 0
    # # 16号--多云
    # x_external_information4[10 * 24 * 6:11 * 24 * 6, 0] = 1
    # # 17号--晴
    # x_external_information4[11 * 24 * 6:12 * 24 * 6, 0] = 1
    # # 18号--小雨
    # x_external_information4[12 * 24 * 6:13 * 24 * 6, 0] = 0
    # # 21号--多云/晴
    # x_external_information4[13 * 24 * 6:14 * 24 * 6, 0] = 1
    # # 22号--晴
    # x_external_information4[14 * 24 * 6:15 * 24 * 6, 0] = 1
    # # 23号--晴
    # x_external_information4[15 * 24 * 6:16 * 24 * 6, 0] = 1
    # # 24号--晴
    # x_external_information4[16 * 24 * 6:17 * 24 * 6, 0] = 1
    # # 25号--多云
    # # x_external_information4[17*24*6:18*24*6, 3] = 1
    # # 28号
    # # x_external_information4[18*24*6:19*24*6, 2] = 1
    # # 除去2号的天气信息，此处应该是开始矩阵大小没设计好，所以需要往后移动144个时间片，然后再移动5个时间片
    # x_external_information4 = x_external_information4[24 * 6:, :]
    #
    # # ——————————————————————————————加入外部信息-闸机数量—————————————————————————————————
    # external_information5 = np.load('./npy/train_data/Two_more_features.npy')
    # x_external_information5 = np.zeros([(N_days - 1) * N_hours * N_time_slice, 81])
    # external_information5 = external_information5[:, 0]
    # for i in range((N_days - 1) * N_hours * N_time_slice):
    #     x_external_information5[i, :] = external_information5
    #
    # # ——————————————————————————————加入早晚高峰、一般高峰、平峰信息————————————————————————————————
    # x_external_information9 = np.zeros([(N_days - 1) * N_hours * N_time_slice, 1])
    # # [平峰、一般高峰、早晚高峰] [0,1,2]
    # for i in range(0, (N_days - 1) * N_hours * N_time_slice, 24 * 6):
    #     # ——————————————————早晚高峰—————————————————————
    #     x_external_information9[i + 39:i + 54, 0] = 2  # 7：30 - 9：00
    #     x_external_information9[i + 102:i + 114, 0] = 2  # 17：00 - 19：00
    #     # ——————————————————高峰—————————————————————————
    #     x_external_information9[i + 33:i + 39, 0] = 1  # 6:30-7:30
    #     x_external_information9[i + 63:i + 70, 0] = 1  # 10:30-11:30
    #     x_external_information9[i + 99:i + 102, 0] = 1  # 16:30-17:30
    #     x_external_information9[i + 114:i + 132, 0] = 1  # 19:00-22:00
    #
    # # ——————————————————————————————对齐外部信息————————————————————————————
    # # 对齐外部信息
    # length = node_data_1.shape[0]
    # x_true_external_information1 = x_external_information1[len(x_external_information1) - length:, :]
    # x_true_external_information2 = x_external_information2[len(x_external_information2) - length:, :]
    # # x_true_external_information3 = x_external_information3[5:, :]
    # x_true_external_information4 = x_external_information4[len(x_external_information4) - length:, :]
    # x_true_external_information5 = x_external_information5[len(x_external_information5) - length:, :]
    # x_true_external_information9 = x_external_information9[len(x_external_information9) - length:, :]

    # ——————————————————————————————SHUFFLE—————————————————————————————————————————————
    indices = np.arange(length)
    # 打乱
    np.random.shuffle(indices)
    xr_train = xr_train[indices]  # learning task: (2302, 81, 10)->(2302, 81, 2)
    xp_train = xp_train[indices]
    xt_train = xt_train[indices]
    xredge_train = xredge_train[indices]
    xpedge_train = xpedge_train[indices]
    xtedge_train = xtedge_train[indices]
    # 1、2、4、9为Embedding, 5为Dense
    # x_train_external_information1 = x_true_external_information1[indices]
    # x_train_external_information2 = x_true_external_information2[indices]
    # # x_true_external_information3 = x_true_external_information3[indices]
    # x_train_external_information4 = x_true_external_information4[indices]
    # x_train_external_information5 = x_true_external_information5[indices]
    # x_train_external_information9 = x_true_external_information9[indices]
    target = target[indices]
    edge_target = edge_target[indices]
    mask_matrix = mask_matrix[indices]
    # ————————————————————————————————构建验证集合(24-25号数据作为验证集)—————————————————————————————————————
    # 构造得有点复杂.....2019.05.22
    node_day_24 = np.load('./npy/train_data/day_24.npy')
    node_day_25 = np.load('./npy/train_data/day_25.npy')
    node_day_18 = np.load('./npy/train_data/raw_node_data.npy')[144 * 12:-144, :, :]
    val_node_data = np.concatenate((node_day_18, node_day_24, node_day_25), axis=0)

    raw_edge_data = np.load('./npy/train_data/raw_edge_data.npy')
    edge_day_24 = raw_edge_data[-144:, :, :]
    edge_day_18 = raw_edge_data[144 * 12:-144, :, :]
    edge_day_25 = np.load('./npy/train_data/day_25_edge.npy')
    val_edge_data = np.concatenate((edge_day_18, edge_day_24, edge_day_25), axis=0)

    # normalization
    val_node_data /= node_scale_ratio
    val_edge_data /= edge_scale_ratio

    # 构建好验证集的样本和标签
    val_node_data, val_node_target = generate_x_y(val_node_data, len_seq1, len_seq2, len_seq3, len_pre)
    val_edge_data, val_edge_target = generate_x_y(val_edge_data, len_seq1, len_seq2, len_seq3, len_pre)

    # tuple-->array
    week, day, hour = zip(*val_node_data)
    val_node_data_1 = np.array(week)
    val_node_data_2 = np.array(day)
    val_node_data_3 = np.array(hour)

    week, day, hour = zip(*val_edge_data)
    val_edge_data_1 = np.array(week)
    val_edge_data_2 = np.array(day)
    val_edge_data_3 = np.array(hour)

    val_node_target = np.array(val_node_target)
    val_edge_target = np.array(val_edge_target)
    val_node_target = np.squeeze(val_node_target, axis=1)
    val_edge_target = np.squeeze(val_edge_target, axis=1)
    # 根据val_edge_target获取val_mask矩阵
    val_mask = val_edge_target == 0
    val_mx = np.ma.array(val_edge_target, mask=val_mask)
    val_mask_matrix = val_mx.mask + 0

    # # 将data切割出recent\period\trend数据
    val_length = val_node_data_1.shape[0]
    xr_val = np.zeros([val_length, N_station, len_seq3 * N_flow])
    xp_val = np.zeros([val_length, N_station, len_seq2 * N_flow])
    xt_val = np.zeros([val_length, N_station, len_seq1 * N_flow])
    xredge_val = np.zeros([val_length, N_station, len_seq3 * N_station])
    xpedge_val = np.zeros([val_length, N_station, len_seq2 * N_station])
    xtedge_val = np.zeros([val_length, N_station, len_seq1 * N_station])
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

    for i in range(val_length):
        for j in range(len_seq3):
            for k in range(81):
                xredge_val[i, :, j * 81 + k] = val_edge_data_3[i, j, :, k]
    for i in range(val_length):
        for j in range(len_seq2):
            for k in range(81):
                xpedge_val[i, :, j * 81 + k] = val_edge_data_2[i, j, :, k]
    for i in range(val_length):
        for j in range(len_seq1):
            for k in range(81):
                xtedge_val[i, :, j * 81 + k] = val_edge_data_1[i, j, :, k]
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
    # # for i in range(144):
    # #     x_val_external_information5[i, :] = t
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
    # # #------------val external information往后移一个时间片------------------
    # # x_val_external_information1 = x_val_external_information1[len(x_val_external_information1) - len(xr_val):, :]
    # # x_val_external_information2 = x_val_external_information2[len(x_val_external_information2) - len(xr_val):, :]
    # # x_val_external_information4 = x_val_external_information4[len(x_val_external_information4) - len(xr_val):, :]
    # # x_val_external_information5 = x_val_external_information5[len(x_val_external_information5) - len(xr_val):, :]
    # # x_val_external_information9 = x_val_external_information9[len(x_val_external_information9) - len(xr_val):, :]

    # 这里开始跑模型，神经网络相关的，重点看这里
    # ——————————————————————————————建立模型—————————————————————————————————————
    model = stresnet(c_conf=(len_seq3, N_flow, N_station), p_conf=(len_seq2, N_flow, N_station),
                     t_conf=(len_seq1, N_flow, N_station),  # 站点流量input
                     c1_conf=(len_seq3, N_station, N_station), p1_conf=(len_seq2, N_station, N_station),
                     t1_conf=(len_seq1, N_station, N_station))  # 这里的unit代表了大体的网络深度
    model.compile(optimizer='adam', loss={'node_logits': my_own_loss_function, 'edge_logits': my_own_loss_function},
                  loss_weights={'node_logits': 1, 'edge_logits': 0.2},  # 两个任务各自的权重
                  metrics=['accuracy'])

    filepath = config.filepath_stresnet
    # 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
    checkpoint = ModelCheckpoint("./log/{epoch:02d}-{node_logits_loss:.8f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    reduce_lr = LearningRateScheduler(scheduler)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='min')
    callbacks_list = [checkpoint, reduce_lr, early_stopping]
    K.set_value(model.optimizer.lr, 0.0001)
    # 输入的东西要注意
    history = model.fit([xr_train, xp_train, xt_train, xredge_train, xpedge_train, xtedge_train],
                        [target, edge_target],
                        validation_data=([xr_val, xp_val, xt_val, xredge_val, xpedge_val, xtedge_val],
                                         [val_node_target, val_edge_target]),
                        batch_size=config.batch_size, epochs=config.epochs, callbacks=callbacks_list)

    pd.DataFrame(columns=['loss'], data=history.history['loss']).to_csv(config.loss_acc_csvFile, index=None)



if __name__ == '__main__':
    # --------------------------------stresnet训练-start---------------------------------------#
    # 直接从这里开始看，程序的入口，加载统计好的流量文件
    Metro_Flow_Matrix = np.load('./npy/train_data/raw_node_data.npy')  # shape=(2448, 81, 2)
    Metro_Edge_Flow_Matrix = np.load('./npy/train_data/raw_edge_data.npy')  # shape=(2448, 81, 81)
    train_stresnet(Metro_Flow_Matrix, Metro_Edge_Flow_Matrix)
    # --------------------------------stresnet训练-end---------------------------------------#
