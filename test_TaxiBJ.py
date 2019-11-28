# -*- coding:utf-8 -*-
import datetime
import numpy as np
from models.resnet_TaxiBj import stresnet_TaxiBJ
from models.LSTM_TaxiBJ import lstm_TaxiBJ
import config
import warnings

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
    truth = truth
    truth = np.around(truth, 0)

    # eval_01_25.py生成的文件，替换这里的文件名字，这方面可以再想一想后处理算法来提升精度
    predict = predict * 30
    predict = np.around(predict, 0)
    # predict[(predict < 10) & (predict > 0)] =2
    predict[predict < 3] = 0  # 玄学2

    # 看第一个loss(mae)
    loss_matrix = np.abs(truth - predict)
    mae = loss_matrix.sum() / (truth.shape[0] * truth.shape[1] * truth.shape[2])

    # 看第二个loss(mape)
    # truth[np.where(truth == 0)] = 0.0001
    loss_matrix2 = np.clip(loss_matrix, 0.001, 3000) / np.clip(truth, 0.001, 3000)
    mape = loss_matrix2.sum() / (truth.shape[0] * truth.shape[1] * truth.shape[2])

    # loss_matrix1 = np.abs(truth - predict1)
    # loss1 = loss_matrix1.sum()/(144*81*2)
    # loss_matrix2 = np.abs(truth - predict2)
    # loss2 = loss_matrix2.sum()/(144*81*2)
    # print(truth)
    mdae = np.median(np.abs(truth - predict))
    print(mae, mape, mdae)
    return mae, mape


def test_stresnet_TaxiBJ(data_week, data_day, data_recent, predict_day, truth_day):
    warnings.filterwarnings("ignore")
    # ——————————————————————————————组织数据———————————————————————————————
    # 由于>25的数量极其少，将>25的值全都默认为25
    # sub_Metro_Flow2[np.where(sub_Metro_Flow2 > 25)] = 25
    out_maximum = 30  # 估摸出站的最大阈值  因为在getMetroFlow函数中已经除了100近似归一化了
    data_week /= out_maximum
    data_day /= out_maximum
    # ——————————————————————建立模型———————————————————————————
    model = stresnet_TaxiBJ(c_conf=(len_seq3, N_flow, N_station), p_conf=(len_seq2, N_flow, N_station),
                            t_conf=(len_seq1, N_flow, N_station), nb_residual_unit=4)  # 对应修改这里,和训练阶段保持一致
    model.load_weights('./log/stresnet/TaxiBJ/163-0.32096881.hdf5')
    # model.load_weights(config.model_weights_stresnet)
    # model.summary()
    xr_test = np.zeros([1, N_station, len_seq3 * N_flow])
    xp_test = np.zeros([1, N_station, len_seq2 * N_flow])
    xt_test = np.zeros([1, N_station, len_seq1 * N_flow])

    sum_of_predictions = 24 * 2 - config.len_seq3
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

        for j in range(len_seq3):
            for k in range(2):
                xr_test[0, :, j * 2 + k] = r[j, :, k]
        for j in range(len_seq2):
            for k in range(2):
                xp_test[0, :, j * 2 + k] = p[j, :, k]
        for j in range(len_seq1):
            for k in range(2):
                xt_test[0, :, j * 2 + k] = t[j, :, k]

        # 对应修改了这里
        ans = model.predict([xr_test, xp_test, xt_test])
        data_recent[i + config.len_seq3, :, :] = ans
    np.save(predict_day, data_recent)

    truth = np.load(truth_day)[:, 0:81, :]
    predict = np.load(predict_day)

    mae_compute(truth, predict)

    print('Testing Done...')


def test_LSTM_TaxiBJ(data_week, data_day, data_recent, predict_day, truth_day):
    warnings.filterwarnings("ignore")
    # ——————————————————————————————组织数据———————————————————————————————
    # 由于>25的数量极其少，将>25的值全都默认为25
    # sub_Metro_Flow2[np.where(sub_Metro_Flow2 > 25)] = 25
    out_maximum = 30  # 估摸出站的最大阈值  因为在getMetroFlow函数中已经除了100近似归一化了
    data_week /= out_maximum
    data_day /= out_maximum
    # ——————————————————————建立模型———————————————————————————
    model = lstm_TaxiBJ(c_conf=(len_seq3, N_flow, N_station), p_conf=(len_seq2, N_flow, N_station),
                        t_conf=(len_seq1, N_flow, N_station))  # 对应修改这里,和训练阶段保持一致
    model.load_weights('./log/LSTM/TaxiBJ/100-0.51149772.hdf5')
    # model.load_weights(config.model_weights_stresnet)
    # model.summary()
    xr_test = np.zeros([1, N_station, len_seq3 * N_flow])
    xp_test = np.zeros([1, N_station, len_seq2 * N_flow])
    xt_test = np.zeros([1, N_station, len_seq1 * N_flow])

    sum_of_predictions = 24 * 2 - config.len_seq3
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

        for j in range(len_seq3):
            for k in range(2):
                xr_test[0, :, j * 2 + k] = r[j, :, k]
        for j in range(len_seq2):
            for k in range(2):
                xp_test[0, :, j * 2 + k] = p[j, :, k]
        for j in range(len_seq1):
            for k in range(2):
                xt_test[0, :, j * 2 + k] = t[j, :, k]

        # 对应修改了这里
        ans = model.predict([xr_test, xp_test, xt_test])
        data_recent[i + config.len_seq3, :, :] = ans
    np.save(predict_day, data_recent)

    truth = np.load(truth_day)[:, 0:81, :]
    predict = np.load(predict_day)

    mae_compute(truth, predict)

    print('Testing Done...')


if __name__ == '__main__':
    # ################stresnet
    # # -------------------------------------- test_TaxiBJ_0402day-start -------------------------------------- #
    # # node_data
    # print('test_TaxiBJ_0402day-start')
    # data_day = np.load('./npy/test_data/taxibj_node_data_day0401.npy')[:, 0:81, :]
    # data_week = np.load('./npy/test_data/taxibj_node_data_day0326.npy')[:, 0:81, :]
    # data_recent = np.zeros([N_hours * 2, N_station, N_flow])
    #
    # # 预测文件
    # predict_day = './npy/mae_compare/predict_day0402.npy'
    #
    # # 真实文件
    # truth_day = './npy/test_data/taxibj_node_data_day0402.npy'
    # test_stresnet_TaxiBJ(data_week, data_day, data_recent, predict_day, truth_day)
    #
    # # -------------------------------------- test_TaxiBJ_0402day-end -------------------------------------- #
    # # -------------------------------------- test_TaxiBJ_0403ay-start -------------------------------------- #
    # # node_data
    # print('test_TaxiBJ_0403day-start')
    # data_day = np.load('./npy/test_data/taxibj_node_data_day0402.npy')[:, 0:81, :]
    # data_week = np.load('./npy/test_data/taxibj_node_data_day0327.npy')[:, 0:81, :]
    # data_recent = np.zeros([N_hours * 2, N_station, N_flow])
    #
    # # 预测文件
    # predict_day = './npy/mae_compare/predict_day0403.npy'
    #
    # # 真实文件
    # truth_day = './npy/test_data/taxibj_node_data_day0403.npy'
    # test_stresnet_TaxiBJ(data_week, data_day, data_recent, predict_day, truth_day)
    #
    # # -------------------------------------- test_TaxiBJ_0405day-end -------------------------------------- #
    # # -------------------------------------- test_TaxiBJ_0404day-start -------------------------------------- #
    # # node_data
    # print('test_TaxiBJ_0404day-start ')
    # data_day = np.load('./npy/test_data/taxibj_node_data_day0403.npy')[:, 0:81, :]
    # data_week = np.load('./npy/test_data/taxibj_node_data_day0328.npy')[:, 0:81, :]
    # data_recent = np.zeros([N_hours * 2, N_station, N_flow])
    #
    # # 预测文件
    # predict_day = './npy/mae_compare/predict_day0404.npy'
    #
    # # 真实文件
    # truth_day = './npy/test_data/taxibj_node_data_day0404.npy'
    # test_stresnet_TaxiBJ(data_week, data_day, data_recent, predict_day, truth_day)
    #
    # # -------------------------------------- test_TaxiBJ_0404day-end -------------------------------------- #
    # # -------------------------------------- test_TaxiBJ_0405day-start -------------------------------------- #
    # # node_data
    # print('test_TaxiBJ_0405day-start')
    # data_day = np.load('./npy/test_data/taxibj_node_data_day0404.npy')[:, 0:81, :]
    # data_week = np.load('./npy/test_data/taxibj_node_data_day0329.npy')[:, 0:81, :]
    # data_recent = np.zeros([N_hours * 2, N_station, N_flow])
    #
    # # 预测文件
    # predict_day = './npy/mae_compare/predict_day0405.npy'
    #
    # # 真实文件
    # truth_day = './npy/test_data/taxibj_node_data_day0405.npy'
    # test_stresnet_TaxiBJ(data_week, data_day, data_recent, predict_day, truth_day)
    #
    # # -------------------------------------- test_TaxiBJ_0405day-end -------------------------------------- #

    ###############LSTM
    # -------------------------------------- test_TaxiBJ_0402day-start -------------------------------------- #
    # node_data
    print('test_LSTM_TaxiBJ_0402day-start')
    data_day = np.load('./npy/test_data/taxibj_node_data_day0401.npy')[:, 0:81, :]
    data_week = np.load('./npy/test_data/taxibj_node_data_day0326.npy')[:, 0:81, :]
    data_recent = np.zeros([N_hours * 2, N_station, N_flow])

    # 预测文件
    predict_day = './npy/mae_compare/predict_LSTM_day0402.npy'

    # 真实文件
    truth_day = './npy/test_data/taxibj_node_data_day0402.npy'
    test_LSTM_TaxiBJ(data_week, data_day, data_recent, predict_day, truth_day)

    # -------------------------------------- test_TaxiBJ_0402day-end -------------------------------------- #
    # -------------------------------------- test_TaxiBJ_0403ay-start -------------------------------------- #
    # node_data
    print('test_LSTM_TaxiBJ_0403day-start')
    data_day = np.load('./npy/test_data/taxibj_node_data_day0402.npy')[:, 0:81, :]
    data_week = np.load('./npy/test_data/taxibj_node_data_day0327.npy')[:, 0:81, :]
    data_recent = np.zeros([N_hours * 2, N_station, N_flow])

    # 预测文件
    predict_day = './npy/mae_compare/predict_LSTM_day0403.npy'

    # 真实文件
    truth_day = './npy/test_data/taxibj_node_data_day0403.npy'
    test_LSTM_TaxiBJ(data_week, data_day, data_recent, predict_day, truth_day)

    # -------------------------------------- test_TaxiBJ_0405day-end -------------------------------------- #
    # -------------------------------------- test_TaxiBJ_0404day-start -------------------------------------- #
    # node_data
    print('test_LSTM_TaxiBJ_0404day-start ')
    data_day = np.load('./npy/test_data/taxibj_node_data_day0403.npy')[:, 0:81, :]
    data_week = np.load('./npy/test_data/taxibj_node_data_day0328.npy')[:, 0:81, :]
    data_recent = np.zeros([N_hours * 2, N_station, N_flow])

    # 预测文件
    predict_day = './npy/mae_compare/predict_LSTM_day0404.npy'

    # 真实文件
    truth_day = './npy/test_data/taxibj_node_data_day0404.npy'
    test_LSTM_TaxiBJ(data_week, data_day, data_recent, predict_day, truth_day)

    # -------------------------------------- test_TaxiBJ_0404day-end -------------------------------------- #
    # -------------------------------------- test_TaxiBJ_0405day-start -------------------------------------- #
    # node_data
    print('test_LSTM_TaxiBJ_0405day-start')
    data_day = np.load('./npy/test_data/taxibj_node_data_day0404.npy')[:, 0:81, :]
    data_week = np.load('./npy/test_data/taxibj_node_data_day0329.npy')[:, 0:81, :]
    data_recent = np.zeros([N_hours * 2, N_station, N_flow])

    # 预测文件
    predict_day = './npy/mae_compare/predict_LSTM_day0405.npy'

    # 真实文件
    truth_day = './npy/test_data/taxibj_node_data_day0405.npy'
    test_LSTM_TaxiBJ(data_week, data_day, data_recent, predict_day, truth_day)

    # -------------------------------------- test_TaxiBJ_0405day-end -------------------------------------- #
    # #################Arima
    # # -------------------------------------- test_TaxiBJ_0402day-start -------------------------------------- #
    # # node_data
    # print('test_Arima_TaxiBJ_0402day-start')
    #
    # # 预测文件
    # predict_day = './npy/mae_compare/predict_arima_TaxiBj_day0402.npy'
    #
    # # 真实文件
    # truth_day = './npy/test_data/taxibj_node_data_day0402.npy'
    #
    # truth = np.load(truth_day)[:, 0:81, :]
    # predict = np.load(predict_day)
    # mae_compute(truth, predict)
    #
    # # -------------------------------------- test_TaxiBJ_0402day-end -------------------------------------- #
    # # -------------------------------------- test_TaxiBJ_0403ay-start -------------------------------------- #
    # # node_data
    # print('test_Arima_TaxiBJ_0403day-start')
    #
    # # 预测文件
    # predict_day = './npy/mae_compare/predict_arima_TaxiBj_day0403.npy'
    #
    # # 真实文件
    # truth_day = './npy/test_data/taxibj_node_data_day0403.npy'
    # truth = np.load(truth_day)[:, 0:81, :]
    # predict = np.load(predict_day)
    # mae_compute(truth, predict)
    #
    # # -------------------------------------- test_TaxiBJ_0405day-end -------------------------------------- #
    # # -------------------------------------- test_TaxiBJ_0404day-start -------------------------------------- #
    # # node_data
    # print('test_Arima_TaxiBJ_0404day-start ')
    #
    # # 预测文件
    # predict_day = './npy/mae_compare/predict_arima_TaxiBj_day0404.npy'
    #
    # # 真实文件
    # truth_day = './npy/test_data/taxibj_node_data_day0404.npy'
    # truth = np.load(truth_day)[:, 0:81, :]
    # predict = np.load(predict_day)
    # mae_compute(truth, predict)
    #
    # # -------------------------------------- test_TaxiBJ_0404day-end -------------------------------------- #
    # # -------------------------------------- test_TaxiBJ_0405day-start -------------------------------------- #
    # # node_data
    # print('test_Arima_TaxiBJ_0405day-start')
    #
    # # 预测文件
    # predict_day = './npy/mae_compare/predict_arima_TaxiBj_day0405.npy'
    #
    # # 真实文件
    # truth_day = './npy/test_data/taxibj_node_data_day0405.npy'
    # truth = np.load(truth_day)[:, 0:81, :]
    # predict = np.load(predict_day)
    # mae_compute(truth, predict)
    #
    # # -------------------------------------- test_TaxiBJ_0405day-end -------------------------------------- #
