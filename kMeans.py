# use numpy to implement k-means algorithm

import numpy as np
import random
import matplotlib.pyplot as plt

x1 = np.random.normal(loc=1, scale=.4, size=50)
y1 = np.random.normal(loc=1, scale=.3, size=50)
x2 = np.random.normal(loc=3, scale=.2, size=40)
y2 = np.random.normal(loc=2, scale=.3, size=40)

plt.scatter(x1, y1, c='r')
plt.scatter(x2, y2, c='b')

x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))

data = np.stack((x, y), axis=1)  # (90, 2)


class kMeans():

    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None

    def distane(self, pt, centroids):
        # compute distance using euclidean
        # return the cluster corresponds to the min distance
        min_dist = 10000
        min_index = -1
        for i in range(len(centroids)):
            curr_dist = np.sum((pt-centroids[i])**2)
            if curr_dist < min_dist:
                min_dist = curr_dist
                min_index = i

        return min_index

    def updata_centroids(self, data, classes):
        for i in range(self.k):
            class_i_data = data[np.where(np.array(classes) == i)]
            self.centroids[i] = np.mean(class_i_data, axis=0)

    def fit(self, data):
        self.centroids = random.sample(data, self.k)
        # iteration
        iter = 0
        while iter < self.max_iter:
            iter += 1
            # a list to store class index
            classes = []
            # iterate all points
            for point in data:
                curr_class = self.distane(point, self.centroids)
                classes.append(curr_class)
            # update centroids
            self.updata_centroids(data, classes)

        return np.array(classes)


km = kMeans(k=2, max_iter=10)
res = km.fit(data)

color = ['r', 'c', 'g', 'k', 'b', 'm', 'y', 'w']

for i in range(data.shape[0]):
    plt.scatter(data[i, 0], data[i, 1], c=color[res[i]], s=40)

