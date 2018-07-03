import numpy as np


class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate):
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        np.random.seed(0)
        self.weights_input_hidden = np.random.rand(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.weights_hidden_output = np.random.rand(hidden_dim, output_dim) / np.sqrt(hidden_dim)

    def predict(self, input_vector):
        hidden_input = np.dot(input_vector, self.weights_input_hidden)
        hidden_output = np.tanh(hidden_input)
        output_layer_input = np.dot(hidden_output, self.weights_hidden_output)
        output_layer_ouput = _softmax(output_layer_input)

        return np.argmax(output_layer_ouput)


def _softmax(input_vector):
    exp_scores = np.exp(input_vector)

    return exp_scores / np.sum(exp_scores, keepdims=True)
