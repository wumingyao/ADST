import pandas as pd
import os
import datetime
import numpy as np
import warnings
import config
import pandas as pd

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


# 通过时间戳返回时间片
def getTemporalInterval(timestamp):
    # input:timestamp = 2019-1-25 09:35:27
    # return: 25, 9, 3
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


# 统计流量矩阵(核心)  PS：用到的是比赛原始数据的这3列,通过底下绿色部分代码得到结果
def getMetroFlow(datetime, stations, status):
    """
    :param datetime:一天的记录的时间戳list
    :param stations:一天的记录的站点idlist
    :param status:一天的记录的站点进出状态list
    :return:归一化后的一天每个时间片的每个站点的每个进出人数矩阵
    """

    # 地铁流量矩阵（24*6, 81, 2）
    metro_flow_matrix = np.zeros((N_hours * N_time_slice, N_station, N_flow))
    for i in range(len(datetime)):
        w, h, s = getTemporalInterval(datetime[i])
        station_id = stations[i]
        state = status[i]
        # idx等同于一天中的第几个时间片(10min划分)
        idx = h * 6 + s
        metro_flow_matrix[idx, station_id, state] += 1  # 统计
    # /100 近似归一化 预处理第一步
    metro_flow_matrix /= 100
    return metro_flow_matrix


def read_filesdata_to_npy(path, raw_node_data_path, raw_edge_data_path):
<<<<<<< HEAD
    """
    :param path: the path of origin files(.csv)
    :param raw_node_data_path: the saving path of raw_node_data file(.npy)
    :param raw_edge_data_path:the saving path of raw_edge_data file(.npy)
    :return:
    """
=======
>>>>>>> 8e01c9a7f2416a424f84bc85569e6b60fd434eca
    Metro_Flow_Matrix = np.zeros([N_days * N_hours * N_time_slice, N_station, N_flow])  # (144,81,2)
    # 一天的地铁过渡流矩阵
    Metro_Edge_Flow_Matrix = np.zeros([N_days * N_hours * N_time_slice, N_station, N_station])  # (144,81,81)
    # ——————————————————————————————读取数据————————————————————————————————————
    for root, dirs, files in os.walk(path):
        day = 0
        # 根据文件名排序
        files.sort(key=lambda x: int(x[-6:-4]))
        for filename in files:
            print('filename=', filename)
            day += 1
            # 将对应的文件读入
            df = pd.read_csv(os.path.join(root, filename))
            # -------------------计算地铁的点流量矩阵----------------------
            # 地铁站ID
            station_ID = df['stationID'].values
            # 进出站标识
            status = df['status'].values
            # 刷卡时间
            time = df['time'].values
            # 该矩阵表示，某时间片（10分钟为一个时间片）某站进出人数近视归一化后的结果
            # 归一化后的一天每个时间片的每个站点的每个进出人数矩阵
            sub_Metro_Flow = getMetroFlow(time, station_ID, status)
            # 将所有天数的sub_Metro_Flow组装起来
            Metro_Flow_Matrix[(day - 1) * N_hours * N_time_slice:day * N_hours * N_time_slice, :, :] = sub_Metro_Flow

            # --------------------计算地铁的边流量矩阵----------------------
            df = df.sort_values(by=['userID', 'time'])
            for index, row in df.iterrows():
                _, hour, segment = getTemporalInterval(row['time'])
                status = int(row['status'])
                # 对于每个乘客：
                # 如果状态为1，说明该乘客从该站进入
                # 否则，从该站离开
                if status == 1:
                    # 进入地铁的站点id
                    start_station = int(row['stationID'])
                else:
                    # 离开地铁的站点id
                    end_station = int(row['stationID'])
                    Metro_Edge_Flow_Matrix[(day - 1) * N_hours * N_time_slice
                                           + hour * N_time_slice
                                           + segment, start_station, end_station] += 1
    np.save(raw_node_data_path, Metro_Flow_Matrix)
    np.save(raw_edge_data_path, Metro_Edge_Flow_Matrix)
<<<<<<< HEAD


if __name__ == '__main__':
    path = './origin_dataSet/Metro_train/'
    read_filesdata_to_npy(path, './npy/train_data/raw_node_data_day.py', './npy/train_data/raw_edge_data_day.py')
=======
>>>>>>> 8e01c9a7f2416a424f84bc85569e6b60fd434eca
