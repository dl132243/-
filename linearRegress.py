import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#读取数据并用最简单的线性模型直接拟合
data_train = pd.read_csv('zhengqi_train.txt',sep = '\t')
data_test = pd.read_csv('zhengqi_test.txt',sep = '\t')
x_train = np.mat(data_train.drop(['target'],axis = 1)) # 特征量矩阵化
y_train = np.mat(data_train.target)
x_test = np.mat(data_test)

#线性回归拟合
def linear(x,y):
    xTx = x.T * x
    w = xTx.I * x.T * y.T
    return w             # 返回参数w

#计算均方差
def error_data(x,y,w):
    y_result = x * w
    y_error = y_result -y.T
    m = y_error.shape[0]
    sum = 0
    for i in range(m):
        sum += y_error[i]**2
    error = float(sum/m)
    return error

w_train = linear(x_train,y_train)   #利用线性回归计算得到的参数w
error = error_data(x_train,y_train,w_train)  # 计算均方差

y_test = x_test * w_train  # 利用训练出来的参数w，预测测试集的目标值， 提交结果得到均方差为3.0

'''
训练集均方差为0.1135，而测试集的均方差为3.0，说明了模型过拟合，处理过拟合方法有：
1. 减少特征量 2. 正则化
首先来分析数据特征量
'''

data_train['oringin'] = 'train'
data_test['oringin'] = 'test'
data_all = pd.concat([data_train,data_test],axis = 0,ignore_index = True) # 把训练集和测试集合并

# 数据预处理，观察训练集和测试集数据分布特征
for column in data_all.columns[0:-2]:
    g = sns.kdeplot(data_all[column][(data_all['oringin'] == 'train')],color = 'red',shade = True)
    g = sns.kdeplot(data_all[column][data_all['oringin'] == 'test'],ax=g,color = 'blue',shade =True)
    g.set_xlabel(column)
    g.set_ylabel('Frequency')
    g =g.legend(['train','test'])
    #plt.show()

# 观察结果发现V5 V11 V17 V22 V27特征量在训练集和测试集数据分布不均，删除特征
x_train_cur = np.delete(x_train,[5,11,17,22,27],axis = 1)
x_test_cur = np.delete(x_test,[5,11,17,22,27],axis = 1)
w_train_cur = linear(x_train_cur,y_train)      #得到删除特征值后的参数w
error_cur = error_data(x_train_cur,y_train,w_train_cur)     #计算删除特征之后的均方差,0.1194

y_test_cur = x_test_cur * w_train_cur    #计算删除特征之后的测试集结果， 提交结果后，得到的均方差为0.1307

#pd.DataFrame(y_test_cur).to_csv('zhengqi_result2.txt',index = False,header = False)

print('Over')





