from network import NeuralNetwork
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def generate_data():
    # np.random.seed(int(time.time()))
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.30)
    buffer = np.array([])
    for index, item in enumerate(X):
        buffer = np.append(buffer, (item[0], item[1], y[index]))

    buffer = buffer.reshape(200, 3)
    np.random.shuffle(buffer)

    return buffer[:, :2], buffer[:, 2:3].flatten().astype(int)


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


train = 160
test = 20
X, y = generate_data()
X_train = X[:train, :]
y_train = y[:train]
X_test = X[train:200-test, :]
y_test = y[train:200-test]
X_validation = X[200-test:200, :]
y_validation = y[200-test:200]

nn = NeuralNetwork(2, 100, 2, 0.003, 0.2)

for i in range(0, 20000):
    nn.train(X_train, y_train)

    if i % 1000 == 0:
        print(f'loss after {i} test: {nn.calculate_loss(X_test, y_test)} train: {nn.calculate_loss(X_train, y_train)}')

print(f'validation loss: {nn.calculate_loss(X_validation, y_validation)}')
visualize(X_test, y_test, nn)
