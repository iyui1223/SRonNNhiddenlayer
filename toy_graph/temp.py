import numpy as np

class Matrix:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adjacent = np.zeros((num_nodes, num_nodes))
        self.weight = np.random.randn(num_nodes, num_nodes) * 0.1  # Initialize with small noise
        self.bias = np.zeros((num_nodes, num_nodes))
        self.values = np.random.randn(num_nodes, 1)  # Node features

    def add_connection(self, i, j):
        self.adjacent[i, j] = 1
        self.adjacent[j, i] = 1

    def delete_connection(self, i, j):
        self.adjacent[i, j] = 0
        self.adjacent[j, i] = 0

    def adj_weight(self):
        return np.matmul(self.adjacent, self.weight)

    def update_weight(self, dweight, learn_rate=0.01):
        self.weight -= learn_rate * dweight

class Math_layer:
    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

class Multiply_layer(Math_layer):
    def forward(self, x, weight):
        self.x = x
        self.y = weight
        return np.matmul(self.x.T, self.y)

    def backward(self, dout):
        dx = np.matmul(dout, self.y.T)
        dy = np.matmul(self.x.T, dout)
        return dx, dy

class Addition_layer(Math_layer):
    def forward(self, bias, x):
        return x + bias

    def backward(self, dout):
        return dout, dout

def MSE_loss(pred, truth):
    return np.mean((pred - truth) ** 2)

def MSE_grad(pred, truth):
    return 2 * (pred - truth) / truth.size  # Derivative of MSE

def main():
    num_nodes = 5
    x = np.array([[1], [2], [3], [4], [5]])  # Node features
    y = np.array([[8], [6], [10], [2], [4]])  # Targets

    m = Matrix(num_nodes)
    m.add_connection(0, 3)
    m.add_connection(1, 4)
    m.add_connection(2, 1)
    m.add_connection(3, 0)
    m.add_connection(4, 2)

    multiply_layer = Multiply_layer()
    add_layer = Addition_layer()

    num_epochs = 10
    for epoch in range(num_epochs):
        weight = m.adj_weight()

        # Forward pass
        y_pred = multiply_layer.forward(m.values, weight)
        y_pred = add_layer.forward(m.bias, y_pred)
        loss = MSE_loss(y_pred, y)
        print(f"Epoch {epoch}, Loss: {loss}")

        # Backward pass
        dloss = MSE_grad(y_pred, y)
        dx, dbias = add_layer.backward(dloss)
        dx, dweight = multiply_layer.backward(dx)

        # Update parameters
        m.update_weight(dweight)
        m.bias -= 0.01 * dbias

if __name__ == "__main__":
    main()
