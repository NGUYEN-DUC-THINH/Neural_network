import numpy as np

class cross_entropy():
    def __init__(self):
        pass
    def f(self,Y, Yhat):
        return np.sum((Y-Yhat)**2)/Y.shape[1]

    def df(self,Y, Yhat):
        return (Yhat - Y )/Y.shape[1]
