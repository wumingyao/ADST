## ADST:Forecasting Metro Flow using Attention based Deep Spatio-Temporal Networks with Multi-task Learning

### package version
* python==3.6.8
* keras==2.24
* tensorflow==1.13.1
* pandas==0.24.2
* statsmodels==0.10.1
* windows10

### ADST-Net.png
* ADST Model network diagram

### test.py
* The script is used to test the model, including the training of ADST model (street net), LSTM and ARIMA model. The model path used by street net and LSTM is modified in config.py
* Run python test.py

### train.py
* This script is used to train models, including ADST model (street net), LSTM and ARIMA model. The model parameters are modified in config.py
* Run python train.py

### train_multi.py
* This script is used to train the ADST multi-step model, and the model parameters are modified in config.py
* Run python train_multi.py

### config.py
* Modify model configuration parameter script

### read_files.py
* Read and save the original data (CSV format) as NPY format

### npy
* train_data:Store NPY files for training
* test_data:Store NPY files required for testing
* mae_compare:NPY files for storing real and predicted values

### train_TaxiBJ.py
* This script uses taxibj data set to train ADST model

### test_TaxiBJ.py
* This script is used to test the ADST model trained with taxibj dataset

### errors.py
* Four indicator scripts. In the interim, mass needs to import pycast package in python2 environment

### models文件夹
* Store various model scripts 


### origin_dataSet
* Store original data file(.csv)
* Baidu cloud disk link：https://pan.baidu.com/s/1bALCLtrxYcGQb9P1zbgFMg 
* Extraction code：rami


### Geoman基线链接
* https://github.com/wumingyao/MyGeoMAN.git


