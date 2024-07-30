import random
from engine import Tensor
import numpy as np


class Layer:
    def __init__(self, n_inputs, n_neurons, activation="linear"):
        self.weights = Tensor(fill_random=True, shape=(n_neurons, n_inputs), name="w")
        self.bias = Tensor(fill_zeros=True, shape=(n_neurons, 1), name="b")
        self.activation = activation

    def __call__(self, X):
        z = self.weights @ X + self.bias
        z.name = "z"
        activations = {"relu": z.relu, "sigmoid": z.sigmoid, "tanh": z.tanh}
        a = activations[self.activation]()
        a.name = "a"
        
        return a
    

    def params(self):
        return list(self.weights.arr) + list(self.bias.arr)
    

X = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], name="X")    
y = Tensor([[6, 15, 24, 33]], name="y")    

class Sequential:
    def __init__(self, layers=()):
        self.layers = layers

    def __call__(self, X):
        for i, layer in enumerate(self.layers):
            if i == 0:
                X = layer(X.T)
            else:
                X = layer(X)
        return X
    
    def params(self):
        res = []

        for layer in self.layers:
            res.extend(list(layer.weights.arr) + list(layer.bias.arr))

        return res
    
    def fit(self, X, y, epochs=1, lr=0.01):
        for epoch in range(epochs):
            y_pred = self.__call__(X)
            loss = y_pred.se(y)
            loss.backward()
        self.loss = loss

model = Sequential(layers=(
    Layer(3, 6, activation="relu"),
    Layer(6, 2, activation="relu"),
    Layer(2, 1, activation="relu")
))

model.fit(X, y)
res = model(X)
loss = model.loss
"""print(model.loss.grad_func())
print(model.loss.prev[0].out)"""
