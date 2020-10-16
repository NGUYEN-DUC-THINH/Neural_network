import numpy as np

class Relu():
    def __init__(self):
        pass

    def f(self,z):
        return np.maximum(0,z)
    
    def df(self,z):
        z[z<=0]= 0
        z[z>0] = 1
        return z
    

class softmax():
    def __init__(self):
        pass

    def f(self,V):
        e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
        Z = e_V / e_V.sum(axis = 0)
        return Z
    def df(self,V):
        pass

