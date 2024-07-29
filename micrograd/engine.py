import numpy as np


class Tensor:
    def __init__(self, seq=[], prev=(), random_init=False, shape=None, op=""):
        if not random_init and not shape:
            self.seq = np.array(seq)
        elif not random_init and shape:
            self.seq = np.zeros(shape)
        else:
            self.seq = np.random.random(shape)

        self.prev = set(prev)
        self.op = op
        self.grad = np.zeros(self.seq.shape)
        self._backward = lambda: None

    def __add__(self, other):
        other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.seq + other.seq, prev=(self, other), op="+")
        return out
    
    def __mul__(self, other):
        other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.seq * other.seq, prev=(self, other), op="*")
        return out
    
    def matmul(self, other):
        other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.dot(self.seq, other.seq), prev=(self, other), op="@")
        return out
    
    def __neg__(self):
        out = Tensor(-self.seq)
        return out
    
    def __sub__(self, other):
        out = self + -other
        return out
    
    def T(self):
        out = Tensor(self.seq.T, prev=(self,), op="T")
        return out
    
    def shape(self):
        out = self.seq.shape
        return out
    
    def __repr__(self):
        return f"Tensor object: {self.seq}"

X = Tensor([[1, 2], [3, 4], [5, 6]])
w = Tensor(random_init=True, shape=(16, 2))
b = Tensor(shape=(16, 1))

res = w.matmul(X.T())
print(res)
print(res.grad)