from Neural_network.Network.Base_network import BSnetwork
from Neural_network.Optimizer import gradient_descent
from Neural_network.Layer.FClayer import fclayer
import sys
import numpy as np
import random
from Neural_network.labels import labels
from Neural_network.Loss import *
from sklearn.utils import shuffle



class network(BSnetwork):

    def __init__(self):
        BSnetwork.__init__(self)
        
    def add(self,layer):
        try:
            self.layers[-1].last = False
        except:
            pass
        self.layers.append(layer)

    def compile(self,loss = 'cross_entropy', optimier = 'gradient_descent'):
        self.loss = getattr(sys.modules[__name__],loss)()
        self.optimizer = getattr(sys.modules[__name__],optimier)()
    
    def predict(self,input):
        output = np.array(input).T
        for layer in self.layers:
            output = layer.feedforward(output)
        result = output.T.tolist()
        return result

    def fit(self, X_train, Y_train ,epochs, n = 10):
        label = labels(Y_train)
        Y_new = label.y_new

        self.dict = label.dict
        for i in range(epochs):
            # dao lon vi tri du lieu
            X_train,Y_new = shuffle(X_train,Y_new)
            output = np.array(X_train).T
            Y = np.array(Y_new).T

            # lan truyen tien
            for layer in self.layers:
                output = layer.feedforward(output)
            loss = self.loss.f(Y,output)
            if i % n == 0:
                print("iter %d , loss: %f" %(i , loss))
           

            # lan truyen nguoc
            E = self.loss.df(Y,output)
            for layer in reversed(self.layers):
                E = layer.backpropagation(E)
            
            # cap nhat wieth va bias
            for layer in self.layers:
               layer.weith = self.optimizer.ugrade(layer.weith,layer.dw)
               layer.bias = self.optimizer.ugrade(layer.bias,layer.db)

        
                
            
