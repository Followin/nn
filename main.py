from network import NeuralNetwork
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import time


def generate_data():
    # np.random.seed(int(time.time()))
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.30)
    return X, y


def visualize(X, y, network):
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    plot_decision_boundary(lambda x: network.predict(x), X, y)
    plt.title("Logistic Regression")


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


X, y = generate_data()
nn = NeuralNetwork(2, 60, 2, 0.01)

for i in range(0, 20000):
    nn.train(X, y)

    if i % 1000 == 0:
        print(f'loss after {i} {nn.calculate_loss(X, y)}')

visualize(X, y, nn)
