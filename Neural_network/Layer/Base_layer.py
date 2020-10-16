from abc import abstractmethod,ABC

class Base_layer(ABC):

    def __init__(self):
        self.input_shape = None
        self.units = None
        self.input = None
        self.z = None
        self.a = None

    

    def get_input(self):
        return self.input
    
    def get_a(self):
        return self.a
    
    def get_z(self):
        return self.z
    
    @abstractmethod
    def feedforward(self):
        pass
    
    @abstractmethod
    def backpropagation(self):
        pass

