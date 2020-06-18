from keras.layers import Dense, CuDNNLSTM,Conv1D,Dropout,concatenate,add
from keras.layers.core import  Lambda,Activation
from keras.models import K,Model,Input
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import numpy as np

#设定为自增长
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)

def create_dataset(dataset, look_back,skip):
    '''
    对数据进行处理
    '''
    dataX,dataX2,dataY = [],[],[]
    #len(dataset)-1 不必要 但是可以避免某些状况下的bug
    for i in range(look_back*skip,len(dataset)-1):
        dataX.append(dataset[(i-look_back):i,:])
        dataY.append(dataset[i, :])
        temp=[]
        for j in range(i-look_back*skip,i,skip):
            temp.append(dataset[j,:])
        dataX2.append(temp)

    TrainX = np.array(dataX)
    TrainX2 = np.array(dataX2)
    TrainY = np.array(dataY)
    return TrainX, TrainX2 , TrainY





def LSTNet(trainX1,trainX2,trainY,config):

    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    conv1 = Conv1D(filters=48, kernel_size=6, strides=1, activation='relu')  # for input1
    # It's a probelm that I can't find any way to use the same Conv1D layer to train the two inputs,
    conv2 = Conv1D(filters=48, kernel_size=6 , strides=1, activation='relu')  # for input2
    conv2.set_weights(conv1.get_weights())  # at least use same weight

    conv1out = conv1(input1)
    lstm1out = CuDNNLSTM(64)(conv1out)
    lstm1out = Dropout(config.dropout)(lstm1out)

    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))
    conv2out = conv2(input2)
    lstm2out = CuDNNLSTM(64)(conv2out)
    lstm2out = Dropout(config.dropout)(lstm2out)

    lstm_out = concatenate([lstm1out,lstm2out])
    output = Dense(trainY.shape[1])(lstm_out)

    #highway  使用Dense模拟AR自回归过程，为预测添加线性成份，同时使输出可以响应输入的尺度变化。
    highway_window = config.highway_window
    #截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window*trainX1.shape[2])))(z)
    z = Dense(trainY.shape[1])(z)

    output = add([output,z])
    output = Activation('sigmoid')(output)
    model = Model(inputs=[input1,input2], outputs=output)

    return  model



def trainModel(trainX1,trainX2,trainY,config):
    '''
    trainX，trainY: 训练LSTM模型所需要的数据
    config:  配置文件
    '''
    model = LSTNet(trainX1,trainX2,trainY,config)
    model.summary()
    model.compile(optimizer=config.optimizer, loss=config.loss_metric)
    model.fit([trainX1,trainX2], trainY, epochs=config.epochs, batch_size=config.lstm_batch_size, verbose=config.verbose,validation_split=0.1)

    return model

#多维归一化
def NormalizeMult(data):
    normalize = np.arange(2*data.shape[1],dtype='float64')
    normalize = normalize.reshape(data.shape[1],2)
    print(normalize.shape)
    for i in range(0,data.shape[1]):
        #第i列
        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])
        normalize[i,0] = listlow
        normalize[i,1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta
    return data,normalize

#多维反归一化
def FNormalizeMult(data,normalize):
    data = np.array(data)
    for i in  range(0,data.shape[1]):
        listlow =  normalize[i,0]
        listhigh = normalize[i,1]
        delta = listhigh - listlow
        if delta != 0:
            for j in range(0,data.shape[0]):
                data[j,i]  =  data[j,i]*delta + listlow
    return data


def startTrainMult(data,name,config):
    '''
    data: 多维数据
    返回训练好的模型
    '''
    data = data.iloc[:,1:]
    print(data.columns)
    yindex = data.columns.get_loc(name)
    data = np.array(data,dtype='float64')

    #数据归一化
    data, normalize = NormalizeMult(data)
    data_y = data[:,yindex]
    data_y = data_y.reshape(data_y.shape[0],1)
    print(data.shape, data_y.shape)

    #构造训练数据
    trainX1,trainX2, _ = create_dataset(data, config.n_predictions,config.skip)
    _ , _,trainY = create_dataset(data_y,config.n_predictions,config.skip)
    print("trainX Y shape is:",trainX1.shape,trainX2.shape,trainY.shape)

    if len(trainY.shape) == 1:
        trainY = trainY.reshape(-1,1)
    # 进行训练
    model = trainModel(trainX1, trainX2 , trainY, config)

    return model,normalize


