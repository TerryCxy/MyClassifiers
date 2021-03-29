from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, req):
    loss = 0.0
    dw = np.zeros_like(W)
    num_class = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        score = np.dot(X[i], W)
        p = np.zeros(score.shape)
        # score=score-np.max(score)
        score = np.array(score, dtype=np.float64)
        correct_score = score[y[i]]
        sum = np.sum(np.exp(score))
        # loss -= correct_score + np.log(sum)
        # for j in range(num_class):
        #     p[j] = np.exp(score[j]) / np.sum(np.exp(score))
        #     if j == y[i]:
        #         dw[:, j] = dw[:, j] + (1 - p[j]) * X[i]
        #     else:
        #         dw[:, j] = dw[:, j] + p[j] * X[i]
    # loss /= num_train
    #loss = loss + req * np.sum(W * W)
    # dw = dw / num_train + np.sum(req * W)
    return loss