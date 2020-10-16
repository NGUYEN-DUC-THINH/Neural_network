from abc import abstractmethod,ABC


class BSnetwork(ABC):

    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.dict = None


    @abstractmethod

    def add(self):
        pass

    def compile(self):
        pass

    def sumary(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass