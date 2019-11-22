import unittest

import numpy as  np

## required modules from pycast
from pycast.errors import MeanAbsoluteScaledError
from pycast.common.timeseries import TimeSeries


def Mae(truth, predict):
    # 后处理计算
    truth = np.around(truth, 0)
    predict = np.around(predict, 0)
    predict[predict < 3] = 0  # 玄学2
    loss_matrix = np.abs(truth - predict)
    mae = loss_matrix.sum() / (truth.shape[0] * truth.shape[1] * truth.shape[2])
    return mae


def Mape(truth, predict):
    # 后处理计算
    truth = np.around(truth, 0)
    predict = np.around(predict, 0)
    loss_matrix = np.abs(truth - predict)
    loss_matrix2 = np.clip(loss_matrix, 0.001, 3000) / np.clip(truth, 0.001, 3000)
    mape = loss_matrix2.sum() / (truth.shape[0] * truth.shape[1] * truth.shape[2])
    return mape


def Made(truth, predict):
    # 后处理计算
    truth = np.around(truth, 0)
    predict = np.around(predict, 0)
    made = np.median(np.abs(truth - predict))
    return made


def Mase(predict, truth):
    predict = predict.reshape([predict.shape[0] * predict.shape[1], predict.shape[2]])
    truth = truth.reshape([truth.shape[0] * truth.shape[1], truth.shape[2]])
    historyLength = 1
    em = MeanAbsoluteScaledError(historyLength=historyLength)
    historyLength += 1
    predict = predict[historyLength:]
    truth = truth[historyLength:]
    differencelist = []
    for orgValue, forValue in zip(predict, truth):
        difference = orgValue[1] - forValue[1]
        difference = abs(difference)
        differencelist.append(em.local_error([orgValue[1]], [forValue[1]]))
        assert difference == em.local_error([orgValue[1]], [forValue[1]])
    return sum(differencelist) / len(differencelist)
