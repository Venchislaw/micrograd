import numpy as np


class Tensor:
    def __init__(self, arr=[], prev=(), op="", fill_random=False, shape=None):
        if len(arr) > 0:
            self.arr = np.asarray(arr)
        elif fill_random:
            assert shape is not None, "Must pass Shape with random fill"
            self.arr = np.random.random(shape)
        else:
            assert shape is not None, "Must pass shape with zeros fill"
            self.arr = np.zeros(shape)

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

        self.check_broadcast(other)

        def grad_func():
            # Note!!! It's not Tensor.dot, it's np.dot
            self.grad += out.grad * other.arr
            other.grad += out.grad * self.arr
        
        out.grad_func = grad_func()
        return out
    
    def dot(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.shape[-1] == other.shape[0]
        out = Tensor(self.arr @ other.arr, prev=(self, other), op="dot")

        self.check_broadcast(other)

        def grad_func():
            # Note!!! It's not Tensor.dot, it's np.dot
            self.grad += out.grad.dot(other.arr.T)
            other.grad += self.arr.T.dot(out.grad)

        out.grad_fun = grad_func()
        return out
    

    def check_broadcast(self, other):
        for n, (i, j) in enumerate(zip(self.shape, other.shape)):
            if i == j:
                continue

            elif i > j:
                other.broadcast_dim = n
            else:
                self.check_broadcast = n


    @property
    def shape(self):
        return self.arr.shape
    
    @property
    def T(self):
        return Tensor(self.arr.T, prev=(self,), op="T")
    
    def __repr__(self):
        return f"Tensor object: {self.arr.shape}"
    
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
b = Tensor(shape=(10, 1))

Xt = X.T
mul = W.dot(Xt)
res = mul + b

res.grad = np.ones(res.shape)
res.grad_func()
mul.grad_func()
print(W.grad)