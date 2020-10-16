from Neural_network.Layer.Base_layer import Base_layer
import numpy as np
import sys
from Neural_network.Activate_functions import *


class fclayer(Base_layer):
    
    def __init__(self,units,act_func,input_shape = None):
        Base_layer.__init__(self)
        self.input_shape = input_shape
        self.units = units
        self.act_func = getattr(sys.modules[__name__], act_func)()
        self.weith = 0.01*np.random.rand(self.input_shape,self.units)
        self.bias = np.zeros((self.units, 1))
        self.dw = None
        self.db = None
        self.last = True

    
    def feedforward(self,input):
        self.input = input
        self.z = self.weith.T @ self.input + self.bias
        self.a = self.act_func.f(self.z)
        return self.a


    def backpropagation(self,W_Eb):
        if self.last:
            E = W_Eb
        else:
            E = W_Eb * self.act_func.df(self.z)
        self.dw = self.input @ E.T
        self.db = np.sum(E, axis = 1, keepdims = True)
        W_E = self.weith @  E 
        return W_E





        

    
