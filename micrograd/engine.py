import numpy as np


class Tensor:
    def __init__(self, arr=[], prev=(), op="", fill_random=False, fill_zeros=False, shape=None):
        if fill_random:
            assert shape is not None, "Must pass Shape with random fill"
            self.arr = np.random.random(shape)
        elif fill_zeros:
            assert shape is not None, "Must pass shape with zeros fill"
            self.arr = np.zeros(shape, dtype=float)
        else:
            self.arr = np.asarray(arr, dtype=float)

        self.prev = prev
        self.op = op
        self.grad = np.zeros(self.arr.shape)
        self.broadcast_dim = None

        self.grad_func = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.arr + other.arr, prev=(self, other), op="+")

        self.check_broadcast(other)

        def grad_func():
            self.grad += out.grad if not self.broadcast_dim else out.grad.sum(axis=self.broadcast_dim, keepdims=True)
            other.grad += out.grad if not other.broadcast_dim else out.grad.sum(axis=other.broadcast_dim, keepdims=True)

        out.grad_func = grad_func
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.arr * other.arr, prev=(self, other), op="*")

        def grad_func():
            # Note!!! It's not Tensor.dot, it's np.dot
            self.grad += out.grad * other.arr
            other.grad += out.grad * self.arr
        
        out.grad_func = grad_func
        return out
    
    def dot(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.shape[-1] == other.shape[0]
        out = Tensor(self.arr @ other.arr, prev=(self, other), op="dot")

        def grad_func():
            # Note!!! It's not Tensor.dot, it's np.dot
            self.grad += out.grad.dot(other.arr.T)
            other.grad += self.arr.T.dot(out.grad)

        out.grad_fun = grad_func
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Support only int or float"

        out = Tensor(self.arr**other, prev=(self, ), op="**{other}")

        def grad_func():
            self.grad += out.grad * (other * self.arr ** (other - 1))
            
        out.grad_func = grad_func
        return out
    

    def relu(self):
        out = Tensor(np.maximum(self.arr, 0), prev=(self,), op="ReLU")

        def grad_func():
            self.grad += out.grad * (self.arr > 0)
        
        out.grad_func = grad_func
        return out
    
    def tanh(self):
        t = (np.exp(2 * self.arr) - 1) / (2 * np.exp(-self.arr))
        out = Tensor(t, prev=(self, ), op="tanh")

        def grad_func():
            self.grad += out.grad * (1 - t**2)
        
        out.grad_func = grad_func
        return out
    
    def sigmoid(self):
        out = Tensor(1 / 1 + np.exp(-self.arr), prev=(self, ), op="sigmoid")

        def grad_func():
            self.grad += out.grad * (out.arr * (1 - out.arr))

        out.grad_func = grad_func
        return out 

    def __neg__(self):
        return Tensor(-self.arr)

    def __sub__(self, other):
        return self + -other
    
    def __truediv__(self, other):
        return self * (other**-1)


    def __rmul__(self, other):
        return self * other
    
    def __radd__(self, other):
        return self + other
    
    
    def check_broadcast(self, other):
        for n, (i, j) in enumerate(zip(self.shape, other.shape)):
            if i == j:
                continue

            elif i > j:
                other.broadcast_dim = n
                break
            else:
                self.broadcast_dim = n
                break


    @property
    def shape(self):
        return self.arr.shape
    
    @property
    def T(self):
        return Tensor(self.arr.T, prev=(self,), op="T")
    
    def __repr__(self):
        return f"Tensor object: {self.arr.shape}"
    
    __array_priority__ = 10000
    
"""
a = Tensor([[1, 2, 3]])
b = Tensor([[2, 3, 4]])
c = a + b
d = Tensor([[1, 2, 3]])
e = c * d
f = Tensor([[1, 2, 3]])
g = f.T
h = e.dot(g)
print(h)"""


X = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
W = Tensor(fill_random=True, shape=(10, 3))
b = Tensor(fill_zeros=True, shape=(10, 1))

Xt = X.T
mul = W.dot(Xt)
res_z = mul + b
res_a = res_z.tanh()

res_a.grad = np.ones(res_a.shape)
res_a.grad_func()
res_z.grad_func()
mul.grad_func()
print(W.grad)

a = Tensor([1, 2, 3])

z = a / Tensor([1, 2, 3])
z.grad_func()

print(z.grad)
