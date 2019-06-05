import numpy as np

def softmax(X):
    # max_x = np.max(X)
    # return np.exp(X - max_x) / (np.exp(X - max_x)).sum()
    expX = np.exp(X)
    return expX / expX.sum()
def training(etha, train_x, train_y,ep_num,params):
    for i in range(ep_num):
        sum = 0.0
        for cur_x,cur_y in zip(train_x,train_y):
            forward_ret = forward_prop(cur_y,cur_x,params)
            sum+= forward_ret['loss']
        loss_avg = sum / train_x.shape[0]
        print(loss_avg)

def relu_activation(X):
    return np.maximum(X, 0)
def relu_deriative(X):
    return 1. * (X > 0)
def forward_prop(y, x, params):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, x.reshape(784, 1)) + b1
    h1 = relu_activation(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = softmax(z2)
    loss = -np.log(h2[int(y)])
    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret
if __name__ == "__main__":
    input_size = 784
    h_rows_size = 50
    num_of_classes = 10
    # Initialize random parameters and inputs
    W1 = np.random.uniform(-0.08, 0.08,[h_rows_size,input_size])
    b1 = np.random.rand(h_rows_size,1)
    W2 = np.random.uniform(-0.08, 0.08,[num_of_classes,h_rows_size])
    b2 = np.random.rand(num_of_classes, 1)
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    #load the text file with the data
    x_train = np.loadtxt("train_x", max_rows=255)
    y_train = np.loadtxt("train_y", max_rows=255)
    #test_x = np.loadtxt("test_x")
    training(0.01,x_train,y_train,10,params)