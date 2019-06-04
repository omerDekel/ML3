import numpy as np

def softmax(X):
    # max_x = np.max(X)
    # return np.exp(X - max_x) / (np.exp(X - max_x)).sum()
    expX = np.exp(X)
    return expX / expX.sum()

def relu_activation(X):
    return np.maximum(X, 0)

def relu_deriative(X):
    return 1. * (X > 0)

def forward_prop(x, y, params):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, x.reshape(784, 1)) + b1
    h1 = relu_activation(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = softmax(z2)
    loss = -np.log(h2[np.argmax(y)])
    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret
if __name__ == "__main__":
    # Initialize random parameters and inputs
    W1 = np.random.rand(2, 2)
    b1 = np.random.rand(2, 1)
    W2 = np.random.rand(1, 2)
    b2 = np.random.rand(1, 1)
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    #load the text file with the data
    x_train = np.loadtxt("train_x", max_rows=1)
    y_train = np.loadtxt("train_y", max_rows=1)
    test_x = np.loadtxt("test_x")