# -*- coding:utf-8 -*-
import datetime
import numpy as np
from models.multiscale_multitask_STResNet import stresnet
from models.STResNet_Multi_Step_Pre import stresnet_multi_step_pre
from models.LSTM import LSTM_Net
import config
import warnings
from errors import Mae, Mape, Made
from models.MyArima import *

warnings.filterwarnings("ignore")
START_HOUR = 0  # 凌晨 [0,1,...,23]外部信息
START_MINUTE = 0  # 第一个分钟片 [0,1,2,3,4,5]外部信息

N_days = config.N_days  # 用了多少天的数据(目前17个工作日)
N_hours = config.N_hours
N_time_slice = config.N_time_slice  # 1小时有6个时间片
N_station = config.N_station  # 81个站点
N_flow = config.N_flow  # 进站 & 出站
len_seq1 = config.len_seq1  # week时间序列长度为2
len_seq2 = config.len_seq2  # day时间序列长度为3
len_seq3 = config.len_seq3  # hour时间序列长度为5
nb_flow = config.nb_flow  # 输入特征


def getTemporalInterval(timestamp):
    timestamp = timestamp.replace('/', '-')
    t = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    weekday = t.weekday()
    hour = t.hour
    minute = t.minute
    if 10 > minute >= 0:
        segment = 0
    elif 20 > minute >= 10:
        segment = 1
    elif 30 > minute >= 20:
        segment = 2
    elif 40 > minute >= 30:
        segment = 3
    elif 50 > minute >= 40:
        segment = 4
    else:
        segment = 5
    return weekday, hour, segment


# datetime, stations, status = [],[],[]
def getMetroFlow(datetime, stations, status):
    # 地铁流量矩阵（24*6, 81, 2）
    metro_flow_matrix = np.zeros((N_hours * N_time_slice, N_station, N_flow))
    for i in range(len(datetime)):
        w, h, s = getTemporalInterval(datetime[i])
        station_id = stations[i]
        state = status[i]
        idx = h * 6 + s
        metro_flow_matrix[idx, station_id, state] += 1
    # /100 近似归一化
    metro_flow_matrix /= 100
    return metro_flow_matrix


def mae_compute(truth, predict):
    # 后处理计算
    truth = truth * 100
    predict = predict * 2945
    # 去掉站点id=54的信息，不用进行比较
    truth2 = np.zeros((144, 80, 2))
    predict2 = np.zeros((144, 80, 2))
    for i in range(truth.shape[1] - 1):
        if i < 53:
            truth2[:, i, :] = truth[:, i, :]
            predict2[:, i, :] = predict[:, i, :]
        elif i > 53:
            truth2[:, i - 1, :] = truth[:, i, :]
            predict2[:, i - 1, :] = predict[:, i, :]

    # 看第一个loss(mae)
    mae = Mae(truth2, predict2)

    # 看第二个loss(mape)
    mape = Mape(truth2, predict2)

    # 第三个评价指标
    mdae = Made(truth2, predict2)

    print(mae, mape, mdae)
    return mae, mape, mdae


def test_stresnet(data_week, data_day, data_recent, data_week_edge, data_day_edge, data_recent_edge, predict_day,
                  truth_day):
    # ——————————————————————————————组织数据———————————————————————————————
    # 由于>25的数量极其少，将>25的值全都默认为25
    # sub_Metro_Flow2[np.where(sub_Metro_Flow2 > 25)] = 25
    out_maximum = 30  # 估摸出站的最大阈值  因为在getMetroFlow函数中已经除了100近似归一化了
    data_week /= out_maximum
    data_day /= out_maximum
    edge_out_maximum = 233.0
    data_week_edge /= edge_out_maximum
    data_day_edge /= edge_out_maximum

    # ——————————————————————建立模型———————————————————————————
    model = stresnet(c_conf=(len_seq3, N_flow, N_station), p_conf=(len_seq2, N_flow, N_station),
                     t_conf=(len_seq1, N_flow, N_station),
                     c1_conf=(len_seq3, N_station, N_station), p1_conf=(len_seq2, N_station, N_station),
                     t1_conf=(len_seq1, N_station, N_station),
                     external_dim1=1, external_dim2=1, external_dim3=None,
                     external_dim4=1, external_dim5=81, external_dim6=None,
                     external_dim7=None, external_dim8=None, external_dim9=1,
                     nb_residual_unit=4, nb_edge_residual_unit=4)  # 对应修改这里,和训练阶段保持一致
    # model.load_weights('log/edge_conv1d/units_12_3channel_external/05-0.00730342.hdf5')  # 替换这里哦，记住记住！
    model.load_weights(config.model_weights_stresnet)
    # model.summary()
    xr_test = np.zeros([1, N_station, len_seq3 * N_flow])
    xp_test = np.zeros([1, N_station, len_seq2 * N_flow])
    xt_test = np.zeros([1, N_station, len_seq1 * N_flow])

    xredge_test = np.zeros([1, N_station, len_seq3 * N_station])
    xpedge_test = np.zeros([1, N_station, len_seq2 * N_station])
    xtedge_test = np.zeros([1, N_station, len_seq1 * N_station])
    # 0125是周五,weekday信息
    x_val_external_information1 = np.zeros([24 * 6, 1])
    x_val_external_information1[:, 0] = 4
    # 时间片信息
    x_val_external_information2 = np.zeros([24 * 6, 1])
    HOUR = 0
    for i in range(0, 24 * 6):
        x_val_external_information2[i, 0] = HOUR
        HOUR = HOUR + 1
    # 天气信息,25号为多云
    x_val_external_information4 = np.zeros([24 * 6, 1])
    x_val_external_information4[:, 0] = 1
    # 闸机信息
    x_val_external_information5 = np.zeros([24 * 6, 81])
    t = np.load('./npy/train_data/Two_more_features.npy')
    t = t[:, 0]
    for i in range(144):
        x_val_external_information5[i, :] = t
    # 早晚高峰、高峰、平峰信息
    x_val_external_information9 = np.zeros([N_hours * N_time_slice, 1])
    # #——————————————————早晚高峰—————————————————————
    x_val_external_information9[39:54, 0] = 2  # 7：30 - 9：00
    x_val_external_information9[102:114, 0] = 2  # 17：00 - 19：00
    # #——————————————————高峰—————————————————————————
    x_val_external_information9[33:39, 0] = 1  # 6:30-7:30
    x_val_external_information9[63:70, 0] = 1  # 10:30-11:30
    x_val_external_information9[99:102, 0] = 1  # 16:30-17:30
    x_val_external_information9[114:132, 0] = 1  # 19:00-22:00

    sum_of_predictions = 24 * 6 - config.len_seq3
    for i in range(sum_of_predictions):
        # if i + 4 + config.len_seq1 >= len(data_week) or i + 3 + config.len_seq2 >= len(
        #         data_day) or i + config.len_seq3 >= len(data_recent):
        #     break
        t = data_week[
            i + config.len_seq3 - config.len_seq1 + 1:i + config.len_seq3 - config.len_seq1 + 1 + config.len_seq1, :,
            :]  # trend
        p = data_day[
            i + config.len_seq3 - config.len_seq2 + 1:i + config.len_seq3 - config.len_seq2 + 1 + config.len_seq2, :,
            :]  # period
        r = data_recent[i:i + config.len_seq3, :, :]  # recent
        et = data_week_edge[
             i + config.len_seq3 - config.len_seq1 + 1:i + config.len_seq3 - config.len_seq1 + 1 + config.len_seq1, :,
             :]
        ep = data_day_edge[
             i + config.len_seq3 - config.len_seq2 + 1:i + config.len_seq3 - config.len_seq2 + 1 + config.len_seq2, :,
             :]
        er = data_recent_edge[i:i + config.len_seq3, :, :]
        mask = np.ones([1, 81, 81])
        for j in range(len_seq3):
            for k in range(2):
                xr_test[0, :, j * 2 + k] = r[j, :, k]
        for j in range(len_seq2):
            for k in range(2):
                xp_test[0, :, j * 2 + k] = p[j, :, k]
        for j in range(len_seq1):
            for k in range(2):
                xt_test[0, :, j * 2 + k] = t[j, :, k]

        for j in range(len_seq3):
            for k in range(81):
                xredge_test[0, :, j * 81 + k] = er[j, :, k]
        for j in range(len_seq2):
            for k in range(81):
                xpedge_test[0, :, j * 81 + k] = ep[j, :, k]
        for j in range(len_seq1):
            for k in range(81):
                xtedge_test[0, :, j * 81 + k] = et[j, :, k]
        # 对应修改了这里
        ans, ans1 = model.predict([xr_test, xp_test, xt_test, xredge_test, xpedge_test, xtedge_test, mask,
                                   x_val_external_information1, x_val_external_information2,
                                   x_val_external_information4,
                                   x_val_external_information5, x_val_external_information9])
        data_recent[i + config.len_seq3, :, :] = ans
    np.save(predict_day, data_recent)

    # truth = np.load(truth_day)
    # predict = np.load(predict_day)
    # mae_compute(truth, predict)

    # print('Testing Done...')


def test_LSTM(data_week, data_day, data_recent, data_week_edge, data_day_edge, data_recent_edge, predict_day,
              truth_day):
    # ——————————————————————————————组织数据———————————————————————————————
    # 由于>25的数量极其少，将>25的值全都默认为25
    # sub_Metro_Flow2[np.where(sub_Metro_Flow2 > 25)] = 25
    out_maximum = 30  # 估摸出站的最大阈值  因为在getMetroFlow函数中已经除了100近似归一化了
    data_week /= out_maximum
    data_day /= out_maximum
    edge_out_maximum = 233.0
    data_week_edge /= edge_out_maximum
    data_day_edge /= edge_out_maximum

    # ——————————————————————建立模型———————————————————————————
    model = LSTM_Net(c_conf=(len_seq3, N_flow, N_station), p_conf=(len_seq2, N_flow, N_station),
                     t_conf=(len_seq1, N_flow, N_station),
                     c1_conf=(len_seq3, N_station, N_station), p1_conf=(len_seq2, N_station, N_station),
                     t1_conf=(len_seq1, N_station, N_station),
                     external_dim1=1, external_dim2=1, external_dim3=None,
                     external_dim4=1, external_dim5=81, external_dim6=None,
                     external_dim7=None, external_dim8=None, external_dim9=1,
                     nb_residual_unit=12, nb_edge_residual_unit=12)  # 对应修改这里,和训练阶段保持一致
    # model.load_weights('log/edge_conv1d/units_12_3channel_external/05-0.00730342.hdf5')  # 替换这里哦，记住记住！
    model.load_weights(config.model_weights_LSTM)
    # model.summary()

    xr_test = np.zeros([1, N_station, len_seq3 * N_flow])
    xp_test = np.zeros([1, N_station, len_seq2 * N_flow])
    xt_test = np.zeros([1, N_station, len_seq1 * N_flow])

    xredge_test = np.zeros([1, N_station, len_seq3 * N_station])
    xpedge_test = np.zeros([1, N_station, len_seq2 * N_station])
    xtedge_test = np.zeros([1, N_station, len_seq1 * N_station])
    # 0125是周五,weekday信息
    x_val_external_information1 = np.zeros([24 * 6, 1])
    x_val_external_information1[:, 0] = 4
    # 时间片信息
    x_val_external_information2 = np.zeros([24 * 6, 1])
    HOUR = 0
    for i in range(0, 24 * 6):
        x_val_external_information2[i, 0] = HOUR
        HOUR = HOUR + 1
    # 天气信息,25号为多云
    x_val_external_information4 = np.zeros([24 * 6, 1])
    x_val_external_information4[:, 0] = 1
    # 闸机信息
    x_val_external_information5 = np.zeros([24 * 6, 81])
    t = np.load('./npy/train_data/Two_more_features.npy')
    t = t[:, 0]
    for i in range(144):
        x_val_external_information5[i, :] = t
    # 早晚高峰、高峰、平峰信息
    x_val_external_information9 = np.zeros([N_hours * N_time_slice, 1])
    # #——————————————————早晚高峰—————————————————————
    x_val_external_information9[39:54, 0] = 2  # 7：30 - 9：00
    x_val_external_information9[102:114, 0] = 2  # 17：00 - 19：00
    # #——————————————————高峰—————————————————————————
    x_val_external_information9[33:39, 0] = 1  # 6:30-7:30
    x_val_external_information9[63:70, 0] = 1  # 10:30-11:30
    x_val_external_information9[99:102, 0] = 1  # 16:30-17:30
    x_val_external_information9[114:132, 0] = 1  # 19:00-22:00

    sum_of_predictions = 24 * 6 - config.len_seq3
    for i in range(sum_of_predictions):
        # if i + 4 + config.len_seq1 >= len(data_week) or i + 3 + config.len_seq2 >= len(
        #         data_day) or i + config.len_seq3 >= len(data_recent):
        #     break
        t = data_week[
            i + config.len_seq3 - config.len_seq1 + 1:i + config.len_seq3 - config.len_seq1 + 1 + config.len_seq1, :,
            :]  # trend
        p = data_day[
            i + config.len_seq3 - config.len_seq2 + 1:i + config.len_seq3 - config.len_seq2 + 1 + config.len_seq2, :,
            :]  # period
        r = data_recent[i:i + config.len_seq3, :, :]  # recent
        et = data_week_edge[
             i + config.len_seq3 - config.len_seq1 + 1:i + config.len_seq3 - config.len_seq1 + 1 + config.len_seq1, :,
             :]
        ep = data_day_edge[
             i + config.len_seq3 - config.len_seq2 + 1:i + config.len_seq3 - config.len_seq2 + 1 + config.len_seq2, :,
             :]
        er = data_recent_edge[i:i + config.len_seq3, :, :]
        mask = np.ones([1, 81, 81])
        for j in range(len_seq3):
            for k in range(2):
                xr_test[0, :, j * 2 + k] = r[j, :, k]
        for j in range(len_seq2):
            for k in range(2):
                xp_test[0, :, j * 2 + k] = p[j, :, k]
        for j in range(len_seq1):
            for k in range(2):
                xt_test[0, :, j * 2 + k] = t[j, :, k]

        for j in range(len_seq3):
            for k in range(81):
                xredge_test[0, :, j * 81 + k] = er[j, :, k]
        for j in range(len_seq2):
            for k in range(81):
                xpedge_test[0, :, j * 81 + k] = ep[j, :, k]
        for j in range(len_seq1):
            for k in range(81):
                xtedge_test[0, :, j * 81 + k] = et[j, :, k]
        # 对应修改了这里
        ans, ans1 = model.predict([xr_test, xp_test, xt_test, xredge_test, xpedge_test, xtedge_test, mask,
                                   x_val_external_information1, x_val_external_information2,
                                   x_val_external_information4,
                                   x_val_external_information5, x_val_external_information9])
        data_recent[i + config.len_seq3, :, :] = ans
    np.save(predict_day, data_recent)


def test_Arima(data):
    # 读数据

    predict_arima = np.zeros(shape=(config.N_hours * config.N_time_slice, config.N_station, config.N_flow))
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
                predict = arima.pred(model, 144)
                predict = predict.reshape(len(predict), 1, 1)
                predict_arima[:, i:i + 1, j:j + 1] = predict
            except Exception:
                continue
    # 后处理
    predict_arima = np.where(predict_arima > 0, predict_arima, 0)
    return predict_arima
    # np.save('./npy/mae_compare/predict_arima_day' + date + '.npy', predict_arima)

    # if __name__ == '__main__':
    # # -------------------------------------- test_stresnet_multi_step-25day-start -------------------------------------- #
    # # node_data
    # day25_node = np.load('./npy/train_data/raw_node_data_25.npy')
    # day25_edge = np.load('./npy/train_data/raw_edge_data_25.npy')
    # data_week_list = [14, 15, 16, 17, 18]
    # data_day_list = [20, 21, 22, 23, 24]
    # # data_recent_list=[]
    # print('test_stresnet_multi_step-25day-start')
    # for i in range(len(data_week_list)):
    #     if i == 0:
    #         data_day = day25_node[144 * (data_day_list[i] - 1):144 * data_day_list[i], :, :]
    #     else:
    #         data_day = np.load('./npy/test_data/start=21_steps_pre_predict_day' + str(data_day_list[i]) + '.npy')
    #     data_week = day25_node[144 * (data_week_list[i] - 1):144 * data_week_list[i], :, :]
    #     data_recent = np.zeros([N_hours * N_time_slice, N_station, N_flow])
    #
    #     # edge_data
    #     data_day_edge = day25_edge[144 * (data_day_list[i] - 1):144 * data_day_list[i], :, :]
    #     data_week_edge = day25_edge[144 * (data_week_list[i] - 1):144 * data_week_list[i], :, :]
    #     data_recent_edge = np.zeros([N_hours * N_time_slice, N_station, N_station])
    #
    #     # 预测文件
    #     predict_day = './npy/test_data/start=21_steps_pre_predict_day' + str(data_day_list[i] + 1) + '.npy'
    #
    #     # 真实文件
    #     truth_day = './npy/mae_compare/truth_day25.npy'
    #     test_stresnet(data_week, data_day, data_recent, data_week_edge, data_day_edge, data_recent_edge, predict_day,
    #                   truth_day)
    #
    # pred25 = np.load('./npy/test_data/start=21_steps_pre_predict_day25.npy')
    # truth25 = np.load('./npy/mae_compare/truth_day25.npy')
    # mae_compute(truth25, pred25)
    # # -------------------------------------- test_stresnet_multi_step-25day-end -------------------------------------- #

    # # -------------------------------------- test_LSTM_multi_step-25day-start -------------------------------------- #
    # # node_data
    # day25_node = np.load('./npy/train_data/raw_node_data_25.npy')
    # day25_edge = np.load('./npy/train_data/raw_edge_data_25.npy')
    # data_week_list = [14, 15, 16, 17, 18]
    # data_day_list = [20, 21, 22, 23, 24]
    # # data_recent_list=[]
    # print('test_stresnet_multi_step-25day-start')
    # for i in range(len(data_week_list)):
    #     if i == 0:
    #         data_day = day25_node[144 * (data_day_list[i] - 1):144 * data_day_list[i], :, :]
    #     else:
    #         data_day = np.load('./npy/test_data/LSTM_start=21_steps_pre_predict_day' + str(data_day_list[i]) + '.npy')
    #     data_week = day25_node[144 * (data_week_list[i] - 1):144 * data_week_list[i], :, :]
    #     data_recent = np.zeros([N_hours * N_time_slice, N_station, N_flow])
    #
    #     # edge_data
    #     data_day_edge = day25_edge[144 * (data_day_list[i] - 1):144 * data_day_list[i], :, :]
    #     data_week_edge = day25_edge[144 * (data_week_list[i] - 1):144 * data_week_list[i], :, :]
    #     data_recent_edge = np.zeros([N_hours * N_time_slice, N_station, N_station])
    #
    #     # 预测文件
    #     predict_day = './npy/test_data/LSTM_start=21_steps_pre_predict_day' + str(data_day_list[i] + 1) + '.npy'
    #
    #     # 真实文件
    #     truth_day = './npy/mae_compare/truth_day25.npy'
    #     test_LSTM(data_week, data_day, data_recent, data_week_edge, data_day_edge, data_recent_edge, predict_day,
    #               truth_day)
    #
    # pred25 = np.load('./npy/test_data/LSTM_start=21_steps_pre_predict_day25.npy')
    # truth25 = np.load('./npy/mae_compare/truth_day25.npy')
    # mae_compute(truth25, pred25)
    # # -------------------------------------- test_LSTM_multi_step-25day-end -------------------------------------- #

    # # -------------------------------------- test_Arima_multi_step-25day-start -------------------------------------- #
    # data_train = np.load('./npy/train_data/raw_node_data_25.npy')[:144 * 20, :, :]
    # predict = np.zeros([144, 81, 2])
    # for i in range(5):
    #     predict = test_Arima(data_train)
    #     data_train = np.concatenate((data_train, predict * 30), axis=0)
    # truth25 = np.load('./npy/mae_compare/truth_day25.npy')
    # mae_compute(truth25, predict)
    # # -------------------------------------- test_Arima_multi_step-25day-end -------------------------------------- #
    # pre = np.load('./npy/mae_compare/predict_day0405.npy') * 30
    # truth = np.load('./npy/test_data/taxibj_node_data_day0405.npy')[:, 0:81, :]
    # # 看第一个loss(mae)
    # mae = Mae(truth, pre)
    #
    # # 看第二个loss(mape)
    # mape = Mape(truth, pre)
    #
    # # 第三个评价指标
    # mdae = Made(truth, pre)
    #
    # print(mae, mape, mdae)

#
pre = np.load('./predict_STRES_25.npy')
truth = np.load('./npy/mae_compare/truth_day25.npy')
mae_compute(truth,pre)
