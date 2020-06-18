import  pandas as pd
import  numpy as np
from  sklearn import  metrics
from  lstm.Predict_Interface import  PredictWithData
from Config import  Config

def GetRMSE(y_hat,y_test):
    sum = np.sqrt(metrics.mean_squared_error(y_test, y_hat))
    return  sum

def GetMAE(y_hat,y_test):
    sum = metrics.mean_absolute_error(y_test, y_hat)
    return  sum

def GetMAPE(y_hat,y_test):
    sum = np.mean(np.abs((y_hat - y_test) / y_test)) * 100
    return sum

def GetMAPE_Order(y_hat,y_test):
    #删除y_test 为0元素
    zero_index = np.where(y_test == 0)
    y_hat = np.delete(y_hat,zero_index[0])
    y_test = np.delete(y_test,zero_index[0])
    sum = np.mean(np.abs((y_hat - y_test) / y_test)) * 100
    return sum

config = Config()
print(config)
path = config.multpath
data = pd.read_csv("./pollution.csv")
data = data.drop(['wnd_dir'], axis = 1)
#选取后20%
data = data.iloc[int(0.8*data.shape[0]):,:]
print("长度为",data.shape[0])
name = config.dimname

normalize = np.load(config.multpath+name+".npy")
loadmodelname = config.multpath+name+".h5"

y_hat,y_test = PredictWithData(data,name,loadmodelname,normalize,config)
y_hat = np.array(y_hat,dtype='float64')
y_test = np.array(y_test,dtype='float64')



print("RMSE为",GetRMSE(y_hat,y_test))
print("MAE为",GetMAE(y_hat,y_test))
#print("MAPE为",GetMAPE(y_hat,y_test))
print("MAPE为",GetMAPE_Order(y_hat,y_test))

np.save(config.multpath+name+"y_hat.npy",y_hat)
np.save(config.multpath+name+"y_test.npy",y_test)
print("结束")
