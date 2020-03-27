import unittest

import numpy as  np
#
# ## required modules from pycast
# # from pycast.errors import MeanAbsoluteScaledError
# # from pycast.common.timeseries import TimeSeries
#
#
# def Mae(truth, predict):
#     # 后处理计算
#     truth = np.around(truth, 0)
#     predict = np.around(predict, 0)
#     predict[predict < 3] = 0  # 玄学2
#     loss_matrix = np.abs(truth - predict)
#     mae = loss_matrix.sum() / (truth.shape[0] * truth.shape[1] * truth.shape[2])
#     return mae
#
#
# def Mape(truth, predict):
#     # 后处理计算
#     truth = np.around(truth, 0)
#     predict = np.around(predict, 0)
#     loss_matrix = np.abs(truth - predict)
#     loss_matrix2 = np.clip(loss_matrix, 0.001, 3000) / np.clip(truth, 0.001, 3000)
#     mape = loss_matrix2.sum() / (truth.shape[0] * truth.shape[1] * truth.shape[2])
#     return mape
#
#
# def Made(truth, predict):
#     # 后处理计算
#     truth = np.around(truth, 0)
#     predict = np.around(predict, 0)
#     made = np.median(np.abs(truth - predict))
#     return made
#
#
# def Mase(predict, truth):
#     predict = predict.reshape([predict.shape[0] * predict.shape[1], predict.shape[2]])
#     truth = truth.reshape([truth.shape[0] * truth.shape[1], truth.shape[2]])
#     historyLength = 1
#     em = MeanAbsoluteScaledError(historyLength=historyLength)
#     historyLength += 1
#     predict = predict[historyLength:]
#     truth = truth[historyLength:]
#     differencelist = []
#     for orgValue, forValue in zip(predict, truth):
#         difference = orgValue[1] - forValue[1]
#         difference = abs(difference)
#         differencelist.append(em.local_error([orgValue[1]], [forValue[1]]))
#         assert difference == em.local_error([orgValue[1]], [forValue[1]])
#     return sum(differencelist) / len(differencelist)


def mae_compute(truth, predict):
    # 后处理计算
    truth = truth * 100
    truth = np.around(truth, 0)

    # eval_01_25.py生成的文件，替换这里的文件名字，这方面可以再想一想后处理算法来提升精度
    predict = predict * 2945
    predict = np.around(predict, 0)
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

    truth2 = truth2.flatten()
    predict2 = predict2.flatten()
    index = (truth2 > 5)
    truth2 = truth2[index]
    predict2 = predict2[index]
    # 看第一个loss(mae)
    loss_matrix = np.abs(truth2 - predict2)
    mae = loss_matrix.sum() / (truth2.size)

    # 看第二个loss(mape)
    # truth[np.where(truth == 0)] = 0.0001
    mape = loss_matrix.sum() / (truth2.size)

    # 第三个评价指标
    mdae = np.median(np.abs(truth2 - predict2))

    print(mae, mape, mdae)
    return mae, mape

if __name__ == '__main__':
    truth = np.load('./npy/mae_compare/truth_day28.npy')
    predict = np.load('./npy/mae_compare/predict_day28.npy')
    mae_compute(truth, predict)
