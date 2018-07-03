from network import NeuralNetwork

nn = NeuralNetwork(2, 2, 2, 0.1)

print(nn.predict([0, 1]))
