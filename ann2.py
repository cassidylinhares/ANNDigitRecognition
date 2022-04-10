import numpy as np
import pandas as pd


def sigmoid(x):
    return (1.0/(1 + np.exp(-x)))


def sigmoid_derivative(x):
    # print(x)
    return (x * (1.0 - x))*10


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(45, 1)
        self.weights2 = np.random.randint(10, size=(40, 1)).astype('float64')
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.prediction = sigmoid(
            np.dot(self.input, self.weights1))  # 40x1 predictions

        # self.output = sigmoid(np.dot(self.prediction, self.weights2))

    def mean_square_error(self):
        np.square(self.prediction - self.y)

    def derivative():
        return

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(
            self.prediction.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(
            self.output), self.weights2.T) * sigmoid_derivative(self.prediction)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == "__main__":
    epochs = 1500
    excel = pd.ExcelFile('data.xlsx')

    # [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    datasetX = pd.read_excel(excel, 'X', header=None)
    datasetY = pd.read_excel(excel, 'Y', header=None)  # [[0], [1], [1], [0]]
    X = np.array(datasetX)
    y = np.array(datasetY)

    nn = NeuralNetwork(X, y)

    for i in range(epochs):
        nn.feedforward()
        print(f"Prediction: {nn.prediction}; Error: {nn.mean_square_error()};")
        nn.backprop()

    print(nn.output.transpose())
