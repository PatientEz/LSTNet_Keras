#使用类实现一个配置文件
class Config:
    def __init__(self):
        self.multpath = './Model/'
        self.dimname = 'pollution'
        self.n_predictions = 30
        #skip层参数
        self.skip = 5
        #AR截取时间步
        self.highway_window = 3
        self.dropout = 0.2
        self.optimizer = 'adam'
        self.loss_metric = 'mse'
        self.lstm_batch_size = 64
        self.verbose = 1
        self.epochs = 300

