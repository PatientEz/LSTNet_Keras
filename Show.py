import  matplotlib.pyplot as plt
import  pandas as pd
import  numpy as np

from Config import  Config

#data = pd.read_csv("./pollution.csv")
# columns = data.columns
# print(columns)
# for i in range(1,len(columns)):
#     plt.plot(data.iloc[:,i])
#     plt.title(columns[i])
#     plt.show()

config = Config()
name = config.dimname

y_hat = np.load("./Model/"+name+"y_hat.npy")
y_test= np.load("./Model/"+name+"y_test.npy")
plt.plot(y_test)
plt.plot(y_hat)
plt.show()

