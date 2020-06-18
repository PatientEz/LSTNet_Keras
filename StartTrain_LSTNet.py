import os
import  pandas as pd
import  numpy as np
from lstm.LSTNet_Interface import  startTrainMult
from Config import Config



config = Config()

path = config.multpath
print(path)
isExists = os.path.exists(path)
if not isExists:
    os.makedirs(path)


data = pd.read_csv("./pollution.csv")
#注:为了演示方便故不使用wnd_dir，其实可以通过代码将其转换为数字序列
data = data.drop(['wnd_dir'], axis = 1)
data = data.iloc[:int(0.8*data.shape[0]),:]
print("长度为",data.shape[0])
name = config.dimname



model,normalize = startTrainMult(data,name,config)
#在某些情况下模型无法直接保存 需要保存权重
model.save_weights(config.multpath+name+".h5")
np.save(config.multpath+name+".npy",normalize)










