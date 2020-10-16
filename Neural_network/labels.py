import numpy as np
import pandas as pd


class labels():
    def __init__(self,y):
        self.y = y
        self.dict = None
        self.y_new = self.__convert_laybels__()
    
    def __convert_laybels__(self): 
        labels = list(set(self.y))
        new_labels = []
        for i in range(len(labels)):
            a = np.zeros(len(labels))
            a[i] = 1
            new_labels.append(a.tolist())
        self.dict = dict(zip(labels, new_labels))
        new_y = list((pd.Series(self.y)).map(self.dict))
        return new_y

