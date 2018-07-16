import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

x = np.linspace(0, 50, 51)
y = np.random.randn(51)

plt.scatter(x, y)

# GBM regressor


def gbm(x, y, M):
    x = x.reshape(-1, 1)
    # var to store predictions
    all_preds = np.zeros((M, len(y)))
    # initialization
    pred = np.mean(y)
    # iterate
    # for each iteration fit the residual
    for i in range(M):
        print 'iteration', i+1
        all_preds[i, :] = pred
        residual = y - pred
        clf = DecisionTreeRegressor(max_depth=1)
        clf = clf.fit(x, residual)
        h = clf.predict(x)
        pred += h

    return all_preds


preds = gbm(x, y, 12)

plt.scatter(x, y)
plt.plot(x, preds[0])

plt.scatter(x, y)
plt.plot(x, preds[9])

# plot to see difference each iteration
fig = plt.figure(figsize=[8, 8])
for i in range(12):
    ax = fig.add_subplot(3, 4, i+1)
    ax.set_title('iteration {}'.format(i+1))
    ax.scatter(x, y, s=2.5)
    ax.plot(x, preds[i], c='r')
