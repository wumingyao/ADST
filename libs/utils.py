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
        start_idx, end_idx = label_start_idx - 6 * 24 - i, label_start_idx - 6 * 24 - i + 1
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
        start_idx, end_idx = label_start_idx - 2 * 6 * 24 - i, label_start_idx - 2 * 6 * 24 - i + 1
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
        start_idx, end_idx = label_start_idx - 6 * 24 * 5 - i, label_start_idx - 6 * 24 * 5 - i + 1
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


def generate_x_y_2channel(train, num_of_days, num_of_hours, num_for_predict):
    '''
    Parameters
    ----------
    train: np.ndarray, shape is (num_of_samples, num_of_stations, num_of_flows)

    num_of_days=3, num_of_hours=3: int
    # 比如预测周二7：00-7：10的数据
    同理num_of_days=3指的是前一天周一{6:40-6:50,6:50-7:00,7:00-7:10},而不是指前三天
    同理num_of_hours=3指的是今天周二{6：30-6：40,6:40-6:50,6:50-7:00}
    num_for_predict=1
    Returns
    ----------
    day_data: np.ndarray, shape is (num_of_samples, num_of_stations, num_of_flows, points_per_hour * num_of_days)

    recent_data: np.ndarray, shape is (num_of_samples, num_of_stations, num_of_flows, points_per_hour * num_of_hours)

    target: np.ndarray, shape is (num_of_samples, num_of_stations, num_for_predict)

    '''
    length = len(train)
    data = []
    # 根据recent\day\week来重新组织数据集
    for i in range(length):
        day = search_day2_data(train, num_of_days, i, num_for_predict)
        recent = search_day1_data(train, num_of_hours, i, num_for_predict)
        if day and recent:
            assert day[1] == recent[1]
            day_data = np.concatenate([train[i: j] for i, j in day[0]], axis=0)
            recent_data = np.concatenate([train[i: j] for i, j in recent[0]], axis=0)
            data.append(((day_data, recent_data), train[day[1][0]: day[1][1]]))
    # ----通过上面的计算,data=(训练值(day, recent),真值)
    features, label = zip(*data)
    return features, label


def generate_x_y_1channel(train, num_of_hours, num_for_predict):
    '''
    Parameters
    ----------
    train: np.ndarray, shape is (num_of_samples, num_of_stations, num_of_flows)

    num_of_days=3, num_of_hours=3: int
    # 比如预测周二7：00-7：10的数据
    同理num_of_days=3指的是前一天周一{6:40-6:50,6:50-7:00,7:00-7:10},而不是指前三天
    同理num_of_hours=3指的是今天周二{6：30-6：40,6:40-6:50,6:50-7:00}
    num_for_predict=1
    Returns
    ----------
    day_data: np.ndarray, shape is (num_of_samples, num_of_stations, num_of_flows, points_per_hour * num_of_days)

    recent_data: np.ndarray, shape is (num_of_samples, num_of_stations, num_of_flows, points_per_hour * num_of_hours)

    target: np.ndarray, shape is (num_of_samples, num_of_stations, num_for_predict)

    '''
    length = len(train)
    data = []
    # 根据recent\day\week来重新组织数据集
    for i in range(length):
        recent = search_day1_data(train, num_of_hours, i, num_for_predict)
        if recent:
            recent_data = np.concatenate([train[i: j] for i, j in recent[0]], axis=0)
            data.append((recent_data, train[recent[1][0]: recent[1][1]]))
    # ----通过上面的计算,data=(训练值(recent),真值)
    features, label = zip(*data)
    return features, label


def generate_x_y_1channel_train(train, num_of_hours, num_for_predict):
    '''
    Parameters
    ----------
    train: np.ndarray, shape is (num_of_samples, num_of_stations, num_of_flows)

    num_of_days=3, num_of_hours=3: int
    # 比如预测周二7：00-7：10的数据
    同理num_of_days=3指的是前一天周一{6:40-6:50,6:50-7:00,7:00-7:10},而不是指前三天
    同理num_of_hours=3指的是今天周二{6：30-6：40,6:40-6:50,6:50-7:00}
    num_for_predict=1
    Returns
    ----------
    day_data: np.ndarray, shape is (num_of_samples, num_of_stations, num_of_flows, points_per_hour * num_of_days)

    recent_data: np.ndarray, shape is (num_of_samples, num_of_stations, num_of_flows, points_per_hour * num_of_hours)

    target: np.ndarray, shape is (num_of_samples, num_of_stations, num_for_predict)

    '''
    length = len(train)
    data = []
    # 根据recent\day\week来重新组织数据集
    for i in range(length):
        recent = search_recent_data(train, num_of_hours, i, num_for_predict)
        if recent:
            recent_data = np.concatenate([train[i: j] for i, j in recent[0]], axis=0)
            data.append((recent_data, train[recent[1][0]: recent[1][1]]))
    # ----通过上面的计算,data=(训练值(recent),真值)
    features, label = zip(*data)
    return features, label
