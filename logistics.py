import numpy as np
from builtins import object
from builtins import range
import math
from past.builtins import xrange


class Logistics_model(object):
    def __init__(self):
        pass

    def sigmoid(self, inX):
        inX = np.array(inX, dtype=np.float32)
        hx = 1.0 / (1.0 + np.exp(-inX))
        return hx

    def train_grad_descent(self, X, y, iters=300, a=0.01, limit=0.01):
        n = X.shape[1]
        m = X.shape[0]
        X = np.mat(X)
        theta = np.ones((n, 1))
        for iter in range(iters):
            error = self.sigmoid(X.dot(theta)) - y
            theta = theta - a * (X.transpose()).dot(error) / m
            loss = self.loss(X, y, theta)
            if np.mean(loss) < limit:
                return theta, loss, iter + 1
        return theta, loss, iter + 1

    def train_gradDescent_regu(self, X, y, iters=300, a=0.01, limit=0.01, lamda=1):
        n = X.shape[1]
        X = np.mat(X)
        m = X.shape[0]
        theta = np.ones((n, 1))
        for iter in range(iters):
            error = self.sigmoid(X.dot(theta)) - y
            for i in range(n):
                if i == 0:
                    theta[i] = theta[i] - a * X[:, 0].T.dot(error) / m
                else:
                    theta[i] = theta[i] * (1 - a * lamda / m) - a * (X[:, i].T).dot(error) / m
            loss = self.loss_regu(X, y, theta, lamda)
            if loss < limit:
                return theta, loss, iter + 1
        return theta, loss, iter + 1

    def predict_twoClass(self, X_test, theta):
        pred = self.sigmoid(X_test.dot(theta))
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return pred

    def loss(self, X, y, theta):
        m = X.shape[0]
        loss = 0.0
        hx = self.sigmoid(X.dot(theta))
        for i in range(y.shape[0]):
            if y[i] == 1:
                loss += -np.log(hx[i])
            elif y[i] == 0:
                loss += -np.log(1 - hx[i])
        loss = loss / m
        return loss

    def loss_regu(self, X, y, theta, lamda):
        m = X.shape[0]
        loss = 0.0
        hx = self.sigmoid(X.dot(theta))
        for i in range(y.shape[0]):
            if y[i] == 1:
                loss += -np.log(hx[i])
            elif y[i] == 0:
                loss += -np.log(1 - hx[i])
        loss = loss / m
        regu = lamda * np.sum(np.square(theta)) / 2 / m
        loss = loss + regu
        return loss
