import numpy as np
import pandas as pd

excel = pd.ExcelFile('data.xlsx')

dataset = pd.read_excel(excel, 'X', header=None)
y = pd.read_excel(excel, 'Y', header=None)

print(len(dataset.T), dataset.shape[1])
neurons = 5
num_layers = 3
y_unique = y[0].unique()
y_unique_len = len(y_unique)


def init_network():
    weights1 = np.random.rand(dataset.shape[1], neurons)  # 45x5
    weights2 = np.random.rand(neurons, y_unique_len)  # 5X10
    weights = [weights1, weights2]

    bias1 = np.random.randn(neurons,)
    bias2 = np.random.randn(y_unique_len,)
    biases = [bias1, bias2]

    return weights, biases

# get the dot product of X and the weights


def dot_prod(X, weight):
    return np.dot(X, weight.T)

# take dot product and apply sigmoid function to get the hidden layer


def sigmoid_activation(activation):
    return 1/(1+np.exp(-activation))


def forward_propogation(weights, biases):
    total_layers = 3  # includes input and output as layers
    layers = []
    dots = []
    for i in range(total_layers-1):
        dot = dot_prod(dataset, weights[i]) + biases[i]
        next_layer = sigmoid_activation(dot)

        layers.append(next_layer)
        dots.append(dot)

        dataset = next_layer
    return dataset, layers, dots


def init_back_prop():
    y2 = np.zeros([len(dataset), y_unique_len])
    y2 = pd.DataFrame(y2)
    for i in range(y_unique_len):
        for j in range(len(y2)):
            if y[0][j] == y_unique[i]:
                y2.iloc[j, i] = 1
            else:
                y2.iloc[j, i] = 0
    y2.head()
    return y2


def back_propogation(dataset, weights, y2, alpha):
    output, layers, dots = forward_propogation(dataset, weights)
    delta = []
    delta.append(y2-layers[-1])
    i = num_layers-2
    while i > 0:
        delta.append(dot_prod(delta[-i], weights[-i])
                     * sigmoid_gradient(dots[-(i+1)]))
        i -= 1
    weights[0] = dot_prod(delta[-(i+1)].T, pd.DataFrame(layers[0])) * alpha

    for i in range(1, len(weights)):


def gradient():
    delta = y2 - a2
    return dot_prod(delta, weights2) * layer*(1-layer)


def sigmoid_gradient(x):
    return sigmoid_activation(x)*(1-sigmoid_activation(x))


def cost(y, y_calc, )
