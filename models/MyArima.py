# -*- coding: utf-8 -*-
from statsmodels.tsa.arima_model import ARIMA
import warnings

warnings.filterwarnings("ignore")


class Arima_Class:
    def __init__(self, arima_para):
        # Define the p, d and q parameters in Arima(p,d,q)(P,D,Q) models
        self.p = arima_para['p']
        self.d = arima_para['d']
        self.q = arima_para['q']

    def fit(self, data):
        """
        :param data: 训练数据,1d
        :param model_save_path: 模型保存路径
        :return: 返回模型
        """
        model = ARIMA(data, (self.p, self.d, self.q)).fit()
        model.summary2()
        # 保存模型
        return model

    def pred(self, loaded, pre_step_len):
        """
        :param model_save_path: 模型文件
        :param pre_step_len: 预测未来几步
        :return:返回预测结果
        """
        # 预测未来pre_step_len个单位
        predictions = loaded.forecast(pre_step_len)
        return predictions[0]
