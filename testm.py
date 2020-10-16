from Neural_network.Network.Network import *

model = network()
model.add(fclayer(units = 10, act_func = 'Relu', input_shape = 2))
model.add(fclayer(units = 2, act_func = 'softmax', input_shape = 10))
model.compile(loss = 'cross_entropy', optimier = 'gradient_descent')

a = [[1,1],[1,0],[0,0]]
b = [1,1,0]
model.fit(a,b,1000)
c = model.predict([[1,0],[0,0]])
d = model.dict

print(c)
print(d)
