import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

excel = pd.ExcelFile('data.xlsx')
data = pd.read_excel(excel, 'data', header=None)
# dataY = pd.read_excel(excel, 'Y', header=None)

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

data_test = data[:m].T
Y_test = data_test[-1]
X_test = data_test[0:n-1]

data_train = data[:m].T
Y_train = data_train[-1]
X_train = data_train[0:n-1]
_, m_train = X_train.shape


def init_params():
    weight1 = np.random.rand(5, 45) - 0.5
    bias1 = np.random.rand(5, 1) - 0.5
    weight2 = np.random.rand(10, 5) - 0.5
    bias2 = np.random.rand(10, 1) - 0.5
    return weight1, bias1, weight2, bias2


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_propagation(weight1, bias1, weight2, bias2, X):
    Z1 = weight1.dot(X) + bias1
    A1 = ReLU(Z1)
    Z2 = weight2.dot(A1) + bias2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def relu_derivative(Z):
    return Z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_propagation(Z1, A1, Z2, A2, weight1, weight2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    derivativeWeight2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = weight2.T.dot(dZ2) * relu_derivative(Z1)
    derivativeWeight1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return derivativeWeight1, db1, derivativeWeight2, db2


def update_params(weight1, bias1, weight2, bias2, derivativeWeight1, db1, derivativeWeight2, db2, alpha):
    weight1 = weight1 - alpha * derivativeWeight1
    bias1 = bias1 - alpha * db1
    weight2 = weight2 - alpha * derivativeWeight2
    bias2 = bias2 - alpha * db2
    return weight1, bias1, weight2, bias2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print("Predictions vs Actual: ", predictions, Y)
    print(f"Sum of Square Error: {sum_square_error(predictions, Y)}")

    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    weight1, bias1, weight2, bias2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(weight1, bias1, weight2, bias2, X)
        derivativeWeight1, db1, derivativeWeight2, db2 = backward_propagation(
            Z1, A1, Z2, A2, weight1, weight2, X, Y)
        weight1, bias1, weight2, bias2 = update_params(
            weight1, bias1, weight2, bias2, derivativeWeight1, db1, derivativeWeight2, db2, alpha)
        if i % 10 == 0:
            print("Epcoh: ", i)
            predictions = get_predictions(A2)
            print(f"Accuracy: {get_accuracy(predictions, Y)}")
    return weight1, bias1, weight2, bias2


def make_predictions(X, weight1, bias1, weight2, bias2):
    _, _, _, A2 = forward_propagation(weight1, bias1, weight2, bias2, X)
    predictions = get_predictions(A2)
    return predictions


def sum_square_error(prediction, label):
    return np.sum(np.square(label-prediction))


def test_prediction(index, weight1, bias1, weight2, bias2):
    prediction = make_predictions(
        X_train[:, index, None], weight1, bias1, weight2, bias2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Actual: ", label)


if __name__ == '__main__':
    epoch = 500
    learn_rate = 0.1
    w1, b1, w2, b2 = gradient_descent(
        X_train, Y_train, learn_rate, epoch)

    print(f"Weights used: {w1} {w2}")

    # test_prediction(0, w1, b1, w2, b2)
    # test_prediction(1, w1, b1, w2, b2)
    # test_prediction(2, w1, b1, w2, b2)
    # test_prediction(3, w1, b1, w2, b2)

    test_prediction_set = make_predictions(X_test, w1, b1, w2, b2)
    get_accuracy(test_prediction_set, Y_test)
