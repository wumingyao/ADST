import unittest

import numpy as  np


## required modules from pycast
# from pycast.errors import MeanAbsoluteScaledError
# from pycast.common.timeseries import TimeSeries
#

def Mae(truth, predict):
    # 后处理计算
    truth = np.around(truth, 0)
    predict = np.around(predict, 0)
    predict[predict < 3] = 0  # 玄学2
    loss_matrix = np.abs(truth - predict)
    mae = loss_matrix.sum() / truth.flatten().size
    return mae


def Mape(truth, predict):
    # 后处理计算
    truth = np.around(truth, 0)
    predict = np.around(predict, 0)
    truth = truth.flatten()
    predict = predict.flatten()
    index = (truth > 5)
    truth = truth[index]
    predict = predict[index]
    loss_matrix = np.abs(truth - predict)
    loss_matrix2 = loss_matrix / truth
    mape = loss_matrix2.sum() / truth.size
    return mape


def Made(truth, predict):
    # 后处理计算
    truth = np.around(truth, 0)
    predict = np.around(predict, 0)
    made = np.median(np.abs(truth - predict))
    return made


truth = np.load('npy/test_data/taxibj_node_data_day0405.npy')[:, 0:81, :]
predict = np.load('npy/mae_compare/predict_stnn_bj_45.npy')

print(Mae(truth, predict), Mape(truth, predict), Made(truth, predict))
