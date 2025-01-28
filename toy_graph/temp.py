import numpy as np

class Matrix:
    def __init__(self, num_nodes):
        self.adjacent = np.zeros(num_nodes, num_nodes)
        self.weight = np.zeros(num_nodes, num_nodes)
        self.bias = np.zeros(num_nodes, num_nodes)
        self.values = np.zeros(num_nodes, num_nodes)

    def add_connection(self):
        pass

    def delete_connection(self):
        pass

    def adj_weight(self):
        weight = np.matmul(self.adjacent.transpose(), self.weight)
        return weight
#        layer = Math_layer
#        layer.forward()
        # calculate del val/ del w gradients as well, too.

    def auto_differentiation(self):
        # bottom up style differentiation

    def update_weight(self):
        pass

class Math_layer:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self):
        pass

    def backward(self):
        pass

class Multiply_layer(Math_layer):
    def forward(self, weight, x):
        self.x = x
        self.y = weight
        self.dout = np.matmul(self.x.transpose(), self.y)
        return self.dout

    def backward(self, dout):
        dx = self.y*dout
        dy = self.x*dout
        return dx, dy

        
class Addition_layer(Math_layer):
    def forward(self, bias, x):
        self.x = x
        self.y = bias
        self.dout = self.x.transpose() + self.y
        return self.dout

    def backward(self, dout):
        dx = dout
        dy = dout
        return dx, dy



    



