import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


class fcnn():

    def __init__(self, train_X, train_y, hid, lr=0.01):
        self.input = train_X.shape[1]
        self.hid = hid
        self.output = train_y.shape[1]
        self.train_X = train_X
        self.train_y = train_y
        self.lr = lr
        # initialise weights
        self.W1 = np.random.randn(self.input, self.hid) / np.sqrt(self.input)
        self.B1 = np.zeros((1, self.hid))
        self.W2 = np.random.randn(self.hid, self.output) / np.sqrt(self.hid)
        self.B2 = np.zeros((1, self.output))

    # active function for the first layer
    def tanh(self, X_fc1):
        X_tanh = np.tanh(X_fc1)

        return X_tanh

    # softmax for the output layer
    # input X_fc2 is the result after assign W2
    def softmax(self, X_fc2):
        X_output = np.exp(X_fc2) / float(np.sum(np.exp(X_fc2)))

        return X_output

    # define loss function
    # y is actual value
    # p-1
    def cross_loss(self, X_output, y):
        index = np.where(y == 1)
        loss = -np.log(X_output[:, index])
        # the loss with derivative
        loss_dev = X_output-y

        return loss, loss_dev

    # forward propagation
    def fit(self, epoch=1):
        # iterate whole data set
        for step in range(epoch):
            print 'step', step
            for i in range(self.train_X.shape[0]):
                # forward propagation
                X_input = self.train_X[i, :].reshape(1, self.train_X.shape[1])
                X_fc1 = np.dot(X_input, self.W1) + self.B1
                X_tanh = self.tanh(X_fc1)
                X_fc2 = np.dot(X_tanh, self.W2) + self.B2
                X_output = self.softmax(X_fc2)

                # calculate loss
                loss, loss_dev = self.cross_loss(X_output, self.train_y[i, :])
                print 'loss', float(loss)

                # according loss to do the back propagation
                delta_output = loss_dev
                delta_input = np.dot(delta_output, self.W2.T) * (1 - np.power(X_tanh, 2))

                # update weights
                self.W2 += -self.lr * np.dot(X_tanh.T, delta_output)
                self.B2 += -self.lr * delta_output
                self.W1 += -self.lr * np.dot(X_input.T, delta_input)
                self.B1 += -self.lr * delta_input

    def predict(self, test_X, test_y):
        result = np.zeros((test_X.shape[0], test_y.shape[1]))
        correct = 0
        for i in range(test_X.shape[0]):
            X_input = test_X[i, :]
            X_fc1 = np.dot(X_input, self.W1) + self.B1
            X_sig = self.tanh(X_fc1)
            X_fc2 = np.dot(X_sig, self.W2) + self.B2
            X_output = self.softmax(X_fc2)
            result[i, :] = X_output
            # calculate precision
            if np.argmax(X_output) == np.argmax(test_y[i, :]):
                correct += 1

        accuracy = float(correct)/test_X.shape[0]
        print 'accuracy', accuracy

        return result


iris = load_iris()
train_X = iris.data
enc = OneHotEncoder()
train_y = enc.fit_transform(iris.target.reshape((150, 1)), iris.target).toarray()

train_X2, test_X2, train_y2, test_y2 = train_test_split(train_X, train_y)

# training ...

fc = fcnn(train_X2, train_y2, hid=10)

fc.fit(epoch=3)

# calculate accuracy

output = fc.predict(test_X2, test_y2)

