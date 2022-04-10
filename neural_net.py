import numpy as np
import pandas as pd

excel = pd.ExcelFile('data.xlsx')

dataset = pd.read_excel(excel, 'X', header=None)
y = pd.read_excel(excel, 'Y', header=None)

# print(len(dataset.T), dataset.shape[1])
# Constants
epochs = 100  # run 20x
neurons = 5
num_layers = 3  # input, hidden, output
num_rows = len(dataset)  # 40
y_unique = y[0].unique()  # [0,1,2,3,4,5,6,7,8,9]
y_unique_len = len(y_unique)  # 10


def weight_calc(X, weight):  # get the dot product of X and the weights
    return np.dot(X, weight)


def init_network():
    weights1 = np.random.rand(dataset.shape[1], neurons)  # 45x5
    weights2 = np.random.rand(neurons, y_unique_len)  # 5X10
    weights = [weights1, weights2]

    bias1 = np.random.randn(neurons,)
    bias2 = np.random.randn(y_unique_len,)
    biases = [bias1, bias2]

    return weights, biases


# take dot product and apply sigmoid function to get the hidden layer
def sigmoid_activation(activation):
    return 1/(1+np.exp(-activation))


def forward_propagation(dataset, weights, biases):
    next_layer = None
    layers = []
    dots = []
    for i in range(num_layers-1):
        dot = weight_calc(dataset, weights[i]) + biases[i]
        next_layer = sigmoid_activation(dot)

        layers.append(next_layer)
        dots.append(dot)

        dataset = next_layer
    return next_layer, layers, dots


def init_back_prop():
    y2 = np.zeros([len(dataset), y_unique_len])
    y2 = pd.DataFrame(y2)
    for i in range(y_unique_len):
        for j in range(len(y2)):
            if y[0][j] == y_unique[i]:
                y2.iloc[j, i] = 1
            else:
                y2.iloc[j, i] = 0
    # y2.head()
    return y2


def sigmoid_gradient(x):
    return sigmoid_activation(x)*(1-sigmoid_activation(x))


def cost_function(y, y_calc):
    log = -np.log(y_calc)*y - np.log(-y_calc) * (1-y)
    sum = np.sum(log)
    return np.sum(sum)/num_rows


def back_propagation(dataset, weights, biases, y2, alpha):
    _, layers, dots = forward_propagation(dataset, weights, biases)
    delta = []
    delta.append(y2-layers[-1])
    i = num_layers-2
    while i > 0:
        delta.append(np.dot(delta[-i], weights[-i].T)
                     * sigmoid_gradient(dots[-(i+1)]))
        i -= 1
    weights[0] = np.dot(delta[-(i+1)].T, dataset) * alpha

    for i in range(1, len(weights)):
        weights[i] = np.dot(delta[-(i+1)].T, pd.DataFrame(layers[0])) * alpha
        weights[i] = np.transpose(weights[i])
    weights[0] = weights[0].T
    _, layers, dots = forward_propagation(dataset, weights, biases)
    cost = cost_function(y2, layers[-1])
    return weights, cost


def train(alpha):
    weights, biases = init_network()
    y2 = init_back_prop()

    # _, layers, dots = forward_propagation(dataset, weights, biases)
    # delta1 = y2 - layers[-1]
    # delta2 = np.dot(delta1, weights[-1].T) * layers[0]*(1-layers[0])
    # w1 = np.dot(delta2.T, pd.DataFrame(layers[0])) * 0.01
    # w2 = np.dot(delta1.T, pd.DataFrame(layers[-1])) * 0.01
    # weights = [w1, w2]
    cost_list = []
    for _ in range(epochs):
        weights, cost = back_propagation(dataset, weights, biases, y2, alpha)
        cost_list.append(cost)

    print(cost_list)
    return weights, biases, y2


def accuracy_function(weights, biases, y2):
    out, _, _ = forward_propagation(dataset, weights, biases)
    accuracy = 0
    for i in range(0, len(out)):
        for j in range(0, len(out[i])):
            if out[i][j] >= 0.5 and y2.iloc[i, j] == 1:
                accuracy += 1
    print('Accuracy: ' + str(accuracy/len(dataset)))


if __name__ == '__main__':
    weights, biases, y2 = train(0.01)
    accuracy_function(weights, biases, y2)
