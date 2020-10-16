class gradient_descent():

    def __init__(self,learning_rate = 0.1):
        self.lr = learning_rate

    def set_lr(self,learning_rate):
        self.lr = learning_rate

    def ugrade(self,F,dF):
        F -= self.lr * dF
        return F