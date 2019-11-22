import unittest

import numpy as  np

## required modules from pycast
from pycast.errors import MeanAbsoluteScaledError
from pycast.common.timeseries import TimeSeries


def Mase(dataOrg, dataFor):
    # 去掉站点id=54的信息，不用进行比较
    truth2 = np.zeros((144, 80, 2))
    predict2 = np.zeros((144, 80, 2))

    for i in range(dataFor.shape[1] - 1):
        if i < 53:
            truth2[:, i, :] = dataFor[:, i, :]
            predict2[:, i, :] = dataOrg[:, i, :]
        elif i > 53:
            truth2[:, i - 1, :] = dataFor[:, i, :]
            predict2[:, i - 1, :] = dataOrg[:, i, :]

    dataOrg = predict2
    dataFor = truth2
    dataOrg = dataOrg.reshape([dataOrg.shape[0] * dataOrg.shape[1], dataOrg.shape[2]])
    dataFor = dataFor.reshape([dataFor.shape[0] * dataFor.shape[1], dataFor.shape[2]])
    historyLength = 1
    em = MeanAbsoluteScaledError(historyLength=historyLength)
    historyLength += 1
    dataOrg = dataOrg[historyLength:]
    dataFor = dataFor[historyLength:]
    differencelist = []
    for orgValue, forValue in zip(dataOrg, dataFor):
        difference = orgValue[1] - forValue[1]
        difference = abs(difference)
        differencelist.append(em.local_error([orgValue[1]], [forValue[1]]))
        assert difference == em.local_error([orgValue[1]], [forValue[1]])
    return sum(differencelist) / len(differencelist)


dataOrg = np.load('./predict_day25_5_steps_pre.npy')
dataFor = np.load('./truth_day25.npy')
print(Mase(dataOrg, dataFor))
