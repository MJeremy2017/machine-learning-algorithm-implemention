import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class muti_reg:

    def __init__(self):
        self.b = None

    def fit(self, train_X, train_y):
        train_X = train_X.reshape(len(train_y), -1)
        train_y = train_y.reshape(len(train_y), 1)
        X = np.insert(train_X, 0, 1, axis=1)
        self.b = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(train_y)  # (x.T*x)-1x.T*y

    def predict(self, test_X):
        test_X = test_X.reshape(test_X.shape[0], -1)
        X = np.insert(test_X, 0, 1, axis=1)
        output = np.dot(X, self.b)

        return output.flatten()


def error(y_true, y_pred):
    n = len(y_true)
    sse = np.sum((y_true-y_pred)**2)
    mse = sse/float(n)
    rmse = np.sqrt(mse)
    sst = np.sum((y_true-np.mean(y_true))**2)
    r_square = 1-sse/sst
    # ad_r_square = 1-(1-r_square**2)*(n-1)/(n-k)
    # print ad_r_square
    print 'SSE:', sse, '\nMSE:', mse, '\nRMSE', rmse, '\nR_Square:', r_square


# uni-variable regression

x = np.linspace(1, 10, 20)
y = 2*x+3+np.random.randn(20)

plt.plot(x, y, 'bo')

mr = muti_reg()
mr.fit(x, y)
y_pred = mr.predict(x)

error(y, y_pred)

# plt.grid()
plt.scatter(x, y)
plt.plot(x, 2*x+3, '-r', linewidth=2, label='actual')
plt.plot(x, y_pred, '--b', label='predict')
plt.legend(loc='upper left')

# multi-linear-regression

state = pd.read_csv('state.csv')
data = state.drop([state.columns[0], 'Murder'], axis=1)
data_y = state.Murder

train_X, test_X, train_y, test_y = train_test_split(np.array(data), np.array(data_y))

mr2 = muti_reg()
mr2.fit(train_X, train_y)
test_pred = mr2.predict(test_X)

error(test_y, test_pred)

# SSE: 34.9067045869
# MSE: 2.68513112207
# RMSE 1.6386369708
# R_Square: 0.658421407881
