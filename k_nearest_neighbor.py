import numpy as np
from builtins import object
from builtins import range
import math
from past.builtins import xrange


class KNN(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, L=1):
        num_train = self.X_train.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train))
        if L == 1:
            # dists = self.get_distance_L1(X)
            dists = dists + np.sum((X), axis=1).reshape(num_test, 1)
            dists = dists - np.sum((self.X_train), axis=1).reshape(1, num_train)
            dists = np.abs(dists)
        elif L == 2:
            # dists = self.get_distance_L2(X)
            dists = dists + np.sum(X ** 2, axis=1).reshape(num_test, 1)
            dists = dists + np.sum((self.X_train ** 2), axis=1).reshape(1, num_train)
            dists = dists - 2 * np.dot(X, self.X_train.T)
            dists = np.power(dists,0.5)
        else:
            raise ValueError("Sorry, Wrong L Number!")

        return self.predict_label(dists, k=k)

    # def get_distance_L1(self, X):
    #     num_train = self.X_train.shape[0]
    #     num_test = X.shape[0]
    #     dists = np.zeros((num_test, num_train))
    #     dists = dists + np.sum((X), axis=1).reshape(num_test, 1)
    #     dists -= np.sum((self.X_train), axis=1).reshape(1, num_train)
    #     dists = np.abs(dists)
    #     return dists

    # def get_distance_L2(self, X):
    #     num_train = self.X_train.shape[0]
    #     num_test = X.shape[0]
    #     dists = np.zeros((num_test, num_train))
    #     dists += np.sum(X ** 2, axis=1).reshape(num_test, 1)
    #     dists += np.sum((self.X_train ** 2), axis=1).reshape(1, num_train)
    #     dists -= 2 * np.dot(X, self.X_train.T)
    #     dists = np.aqrt(dists)
    #     return dists

    def predict_label(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            cloest_y = []
            sort = dists[i].argsort()
            cloest_y = self.y_train[sort[0:k]]
            pass
            y_pred[i] = np.argmax(np.bincount(cloest_y))

        return y_pred
