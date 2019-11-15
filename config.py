import os

epochs = 2
batch_size = 64

N_days = 17  # 用了多少天的数据(目前17个工作日)
N_hours = 24
N_time_slice = 6  # 1小时有6个时间片
N_station = 81  # 81个站点
N_flow = 2  # 进站 & 出站
len_seq1 = 2  # week时间序列长度为2
len_seq2 = 3  # day时间序列长度为3
len_seq3 = 5  # hour时间序列长度为5
len_pre = 1  # 预测步长
nb_flow = 2  # 输入特征

# stresnet相关配置
best_weigth_stresnet = '119-0.00385043.hdf5'  # 最好地stresnet模型参数文件
path_stresnet = "./log/stresnet/" + str(len_seq1) + "_" + str(len_seq2) + "_" + str(
    len_seq3)
model_weights_stresnet = path_stresnet + '/' + best_weigth_stresnet  # 模型参数保存路径
filepath_stresnet = path_stresnet + "/{epoch:02d}-{node_logits_loss:.8f}.hdf5"  # stresnet模型参数文件保存路径
loss_acc_csvFile = path_stresnet + '/history.csv'
if not os.path.exists(path_stresnet):
    os.makedirs(path_stresnet)

# stresnet_multi_step_pre相关配置
best_weigth_stresnet_multi_step_pre = '268-0.01172677.hdf5'
path_stresnet_multi_step_pre = "./log/stresnet/" + str(len_seq1) + "_" + str(len_seq2) + "_" + str(
    len_seq3) + "/MultiPre/"
model_weights_stresnet_multi_step_pre = path_stresnet_multi_step_pre + "/" + best_weigth_stresnet_multi_step_pre
pre_step=5 #预测步长

# LSTM相关配置
best_weigth_LSTM = '182-0.00573374_2_3_5_12units.hdf5'  # 最好的LSTM模型参数文件
path_LSTM = "./log/LSTM/" + str(len_seq1) + "_" + str(len_seq2) + "_" + str(
    len_seq3)
model_weights_LSTM = path_LSTM + '/' + best_weigth_LSTM  # 模型参数保存路径
filepath_LSTM = path_LSTM + "/{epoch:02d}-{node_logits_loss:.8f}.hdf5"  # LSTM模型参数文件保存路径
loss_acc_csvFile = path_LSTM + '/history.csv'
if not os.path.exists(path_LSTM):
    os.makedirs(path_LSTM)
