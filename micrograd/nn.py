import random
from engine import Tensor


class Layer:
    def __init__(self, n_inputs, n_neurons, activation="linear"):
        self.weights = Tensor(fill_random=True, shape=(n_neurons, n_inputs))
        self.bias = Tensor(fill_zeros=True, shape=(n_neurons, 1))
        self.activation = activation

    def __call__(self, X):
        z = self.weights @ X + self.bias
        activations = {"relu": z.relu, "sigmoid": z.sigmoid, "tanh": z.tanh}
        a = activations[self.activation]()
        
        return a
    

    def params(self):
        return list(self.weights.arr) + list(self.bias.arr)
    

X = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])    


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
    
model = Sequential(layers=(
    Layer(3, 6, activation="relu"),
    Layer(6, 2, activation="relu"),
    Layer(2, 1, activation="sigmoid")
))

res = model(X)
print(res.arr)
