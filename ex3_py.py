import numpy as np
import scipy as sp


def softmax(X):
    # max_x = np.max(X)
    # return np.exp(X - max_x) / (np.exp(X - max_x)).sum()
    expX = np.exp(X)
    return expX / expX.sum()


def softmax_derivative(X):
  """ x = softmax(X)
    dx = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                dx[i,j] = x[i] * (1-x[i])
            else:
                 dx[i,j] = -x[i]*x[j]
   # softmax(X)
  #  return dx
                 """""

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

def forward_prop(x, y, params):
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

def back_prop(fprop_cache):
  # Follows procedure given in notes
  x, y, z1, h1, z2, h2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]
  dz2 = (h2 - y)                                #  dL/dz2
  dW2 = np.dot(dz2, h1.T)                       #  dL/dz2 * dz2/dw2

  db2 = dz2                                     #  dL/dz2 * dz2/db2 =  dL/dz2
  dz1 = np.dot(fprop_cache['W2'].T, (h2 - y)) * relu_deriative(z1) #  dL/dz2 * dz2/dh1 * dh1/dz1

  dW1 = np.dot(dz1, x.T)                        #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
  db1 = dz1                                     #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
  return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}


def get_classification(Y):
    return np.argmax(Y)


def convart_classificaion_one_hot(classifications):
    y_new = list()
    size = 10
    for y in classifications:
        new_vec = np.zeros(size)
        new_vec[y] = 1
        y_new.append(new_vec)
    return y_new


if __name__ == "__main__":
    # Initialize random parameters and inputs
    W1 = np.random.rand(2, 2)
    b1 = np.random.rand(2, 1)
    W2 = np.random.rand(1, 2)
    b2 = np.random.rand(1, 1)
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    epoch = 10
    #load the text file with the data
    x_train = np.loadtxt("train_x", max_rows=255)/255.0
    y_train = np.loadtxt("train_y", max_rows=255)
    test_x = np.loadtxt("test_x")
    training(0.01, x_train, y_train, 10, params)


