import numpy as np
from enum import Enum

class InitType(Enum):
    NORMAL = 1
    ZERO = 2
    RANDOM = 3
    EPSILON = 4

class Model:
    def __init__(self, learning_rate, init_type=InitType.RANDOM, epsilon=0.01):
        self.learning_rate = learning_rate
        if init_type == InitType.NORMAL:
            self.theta = np.random.normal(10, 785) * epsilon
        elif init_type == InitType.ZERO:
            self.theta = np.zeros(10, 785)
        elif init_type == InitType.RANDOM:
            self.theta = np.random.rand(10, 785) * epsilon
        elif init_type == InitType.EPSILON:
            self.theta = np.ones(10, 785) * epsilon


    def __hypothesis(self, imgs): # h_theta of x (softmax)
        raw = np.exp(self.theta.dot(imgs))
        return raw / np.repeat(np.sum(raw, axis=0).reshape(1, raw.shape[1]), 10, axis=0)

    def __loss(self, imgs, lbls_logits): # J of theta
        predictions = self.__hypothesis(imgs)
        return -np.sum(np.log(predictions) * lbls_logits)

    def __gradient(self, imgs, lbls_logits):
        predictions = self.__hypothesis(imgs)
        logit_error = np.swapaxes(np.repeat((lbls_logits - predictions).reshape(1, 10, lbls_logits.shape[1]), 785, axis=0), 0, 1)
        imgs_expand = np.repeat(imgs.reshape(1, 785, imgs.shape[1]), 10, axis=0)
        return -np.sum((logit_error * imgs_expand), axis=2)

    def minimize(self, imgs, lbls_logits):
        self.theta -= self.learning_rate * self.__gradient(imgs, lbls_logits)

    def predict(self, imgs):
        return self.__hypothesis(imgs)