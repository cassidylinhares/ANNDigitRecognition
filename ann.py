import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_cost_epoch(cost, epoch):
    plt.plot(epoch, cost)
    plt.xlabel('Sum Square Error')
    plt.ylabel('Epoch')

    plt.show()


def sigmoid(x):
    return (1.0/(1 + np.exp(-x)))


def sigmoid_derivative(x):
    # print(x)
    return (x * (1.0 - x))


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return x > 0


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class NeuralNetwork:
    def __init__(self, x, y, neurons_x, neurons_hidden, neurons_y, epoch, alpha):
        self.input = x.T  # 40x45 -> 45x40
        self.y = y.T  # 40x10 -> 10x40

        self.neurons_x = neurons_x
        self.neurons_hidden = neurons_hidden
        self.neurons_y = neurons_y

        self.learning_rate = alpha
        self.epoch = epoch

        self.weights1 = np.random.rand(neurons_hidden, neurons_x)-0.3  # 5x45
        self.weights2 = np.random.rand(neurons_y, neurons_hidden)-0.3  # 10x5

        self.bias1 = np.zeros((neurons_hidden, 1))
        self.bias2 = np.zeros((neurons_y, 1))

    # make predictions using inner product
    def forward_propagation(self, input):
        # 5x45.45x40 or 5x45.
        self.z1 = np.dot(self.weights1, input) + self.bias1
        self.a1 = relu(self.z1)  # 5x40

        self.z2 = np.dot(self.weights2, self.a1) + self.bias2  # 10x5.5x40
        self.a2 = softmax(self.z2)  # 10x40

    def cost_function(self):
        return -(1/self.y.shape[1]) * np.sum(self.y * np.log(self.a2))

    def sum_square_cost(self):
        return np.sum(np.square(self.y-self.a2))

    # d is the derivative aka gradient to find local min by outputting the amount to step (gradient) and direction
    def backward_propagation(self):
        observations = 1/self.input.shape[1]  # 40

        d_z2 = self.a2 - self.y  # 10x40
        self.d_weights2 = observations * np.dot(d_z2, self.a1.T)  # 10x40.40x5
        self.d_bias2 = observations * \
            np.sum(d_z2, axis=1, keepdims=True)  # sum x1

        d_z1 = observations * \
            np.dot(self.d_weights2.T, d_z2) * \
            relu_derivative(self.a1)  # 5x10.10x40
        self.d_weights1 = observations * \
            np.dot(d_z1, self.input.T)  # 5x40.40x45
        self.d_bias1 = observations * \
            np.sum(d_z1, axis=1, keepdims=True)  # sum x1

    def update_weights_bias(self):
        self.weights1 = self.weights1 - (self.d_weights1 * self.learning_rate)
        self.bias1 = self.bias1 - (self.d_bias1 * self.learning_rate)
        self.weights2 = self.weights2 - (self.d_weights2 * self.learning_rate)
        self.bias2 = self.bias2 - (self.d_bias2 * self.learning_rate)

    # get predictions (forward) find error and adjust weights (back prop) and repeat
    def model(self, input):
        cost_list = []
        sse_list = []
        for i in range(self.epoch):
            self.forward_propagation(input)
            cost = self.cost_function()
            sse = self.sum_square_cost()
            self.backward_propagation()
            self.update_weights_bias()

            cost_list.append(cost)
            sse_list.append(sse)
            if i % 10 == 0:
                print(f"Cost at Epoch {i}: {cost}")

        return cost_list, sse_list

    # get accuracy of predictions
    def accuracy_function(self, input, output):
        self.forward_propagation(input)
        predictions = np.argmax(self.a2, 0)
        actual = np.argmax(output, 0)

        self.accuracy = np.mean(predictions == actual)*100


if __name__ == "__main__":
    # open data
    excel = pd.ExcelFile('data.xlsx')

    x_train = pd.read_excel(excel, 'x_train', header=None)
    y_train = pd.read_excel(excel, 'y_train', header=None)

    x_test = pd.read_excel(excel, 'x_test', header=None)
    y_test = pd.read_excel(excel, 'y_test', header=None)

    X_train = np.array(x_train)
    Y_train = np.array(y_train)

    X_test = np.array(x_test).T
    Y_test = np.array(y_test).T

    epoch = 3000
    e = [i for i in range(1, epoch+1)]

    # define and init network
    nn = NeuralNetwork(
        X_train,
        Y_train,
        neurons_x=45,
        neurons_hidden=5,
        neurons_y=10,
        epoch=epoch,
        alpha=0.06
    )

    _, sse = nn.model(nn.input)

    nn.accuracy_function(nn.input, nn.y)
    print(f"Training Accuracy: {nn.accuracy}%")

    nn.accuracy_function(X_test, Y_test)
    print(f"Test Accuracy: {nn.accuracy}%")

    # index = random.randrange(0, X_test.shape[1])
    # plt.imshow(X_train.T[:, 9].reshape(9, 5))
    # plt.show()

    plot_cost_epoch(sse, e)
