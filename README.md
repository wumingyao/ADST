## ADST:Forecasting Metro Flow using Attention based Deep Spatio-Temporal Networks with Multi-task Learning

### package version
* python==3.6.8
* keras==2.24
* tensorflow==1.13.1
* pandas==0.24.2
* statsmodels==0.10.1
* windows10

### test.py
* 该脚本用于测试模型，分别是ADST模型(stresnet)、LSTM、arima模型的训练，stresnet和LSTM所使用的模型路径在config.py里面修改
* 执行python test.py

### train.py
* 该脚本用于训练模型，分别是ADST模型(stresnet)、LSTM、arima模型的训练，模型参数对应在config.py里面修改
* 执行python train.py

### config.py
* 修改模型配置参数脚本

### read_files.py
* 将原始数据（csv格式）读取并保存为npy格式

### npy
* train_data:存放训练所需要的npy文件
* test_data:存放测试所需要的npy文件
* mae_compare:存放真实值与预测值的npy文件

### models文件夹
* 存放各类模型脚本

<<<<<<< HEAD
### origin_dataSet
* 存放原始数据文件(.csv)
* 百度云网盘链接：https://pan.baidu.com/s/1bALCLtrxYcGQb9P1zbgFMg 
* 提取码：rami

=======
>>>>>>> 8e01c9a7f2416a424f84bc85569e6b60fd434eca
### Geoman文件夹
* 数据预处理已完成，处理模型的时候出现维度不匹配的错误，尚未解决


