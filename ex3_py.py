import numpy as np


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

def training(etha, train_x, train_y,ep_num,params,dev_x,dev_y):
    rnd = np.arange(train_x.shape[0])
    for i in range(ep_num):
        sum = 0.0
        #shuffle
        np.random.shuffle(rnd)
        train_x = train_x[rnd]
        train_y = train_y[rnd]
        for cur_x,cur_y in zip(train_x,train_y):
            forward_ret = forward_prop(cur_x,cur_y,params)
            back_ret = back_prop(forward_ret)
            sum+= forward_ret['loss']
            params = update_param(forward_ret, back_ret, etha)
        loss_avg = sum / train_x.shape[0]
        print("loss avg ",loss_avg)
        predict_on_dev(params,dev_x,dev_y)

    return params


def predict_on_dev(params, dev_x, dev_y):
 sum_loss = good = 0.0 # good counts how many times we were correct
 for x, y in zip(dev_x, dev_y):
    out = forward_prop(x, y, params) # get probabilities vector as result, where index y is the probability that x is classifiedas tag y
    loss = out['loss']
    sum_loss += loss
    if out['h2'].argmax() == y: # model was correct
        good += 1
 acc = good / dev_x.shape[0] # how many times we were correct / # of examples
 avg_loss = sum_loss / dev_x.shape[0] # avg. loss
 print("avg loss, acc",avg_loss, acc)
 #return avg_loss, acc


def testing(test_x, params):
    f = open("test_y", "w")
    for x in test_x:
        out = forward(params, x)
        f.write(str(out.argmax())+'\n')
    f.close()

def forward(params,x):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, x.reshape(784, 1)) + b1
    h1 = relu_activation(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = softmax(z2)
    return h2

def update_param(old_params, grad_params, eta):
    W1, b1, W2, b2 = [old_params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    W1_new, b1_new, W2_new, b2_new = [grad_params[key] for key in ('W1', 'b1', 'W2', 'b2')]

    W1 = W1 - eta * W1_new
    b1 = b1 - eta * b1_new
    W2 = W2 - eta * W2_new
    b2 = b2 - eta * b2_new
    return {'b1': b1, 'W1': W1, 'b2': b2, 'W2': W2}


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
  # vec_y = np.array([0]*10)
  # vec_y[int(y)] = 1
  vec_y = np.reshape(np.zeros(10), (10, 1))
  vec_y[int(y)] = 1
  dz2 = (h2 - vec_y)                                #  dL/dz2

  dW2 = np.dot(dz2, h1.T)                       #  dL/dz2 * dz2/dw2

  db2 = dz2                                     #  dL/dz2 * dz2/db2 =  dL/dz2
  dz1 = np.dot(fprop_cache['W2'].T, (h2 - vec_y)) * relu_deriative(z1) #  dL/dz2 * dz2/dh1 * dh1/dz1

  dW1 = np.dot(dz1, x.T.reshape(1, 784))                        #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
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
    input_size = 784
    h_rows_size = 50
    num_of_classes = 10
    # Initialize random parameters and inputs
    W1 = np.random.uniform(-0.08, 0.08, [h_rows_size, input_size])
    b1 = np.random.rand(h_rows_size, 1)
    W2 = np.random.uniform(-0.08, 0.08, [num_of_classes, h_rows_size])
    b2 = np.random.rand(num_of_classes, 1)
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    epoch = 10
    #load the text file with the data
    x_train = np.loadtxt("train_x", max_rows=5000)/255.0
    y_train = np.loadtxt("train_y", max_rows=5000)
    test_x = np.loadtxt("test_x")

    dev_size = (int)(x_train.shape[0]*0.2)
    dev_x,dev_y = x_train[-dev_size:, :], y_train[-dev_size:]
    x_train, y_train = x_train[:dev_size,:], y_train[:-dev_size]

    params = training(0.01, x_train, y_train, 10, params, dev_x, dev_y)
    testing(test_x,params)


