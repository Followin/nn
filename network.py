import numpy as np


# input: 1 x 2
# w_i_h 2 x 3
# w_h_o 3 x 2

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate):
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        np.random.seed(0)
        self.weights_input_hidden = np.random.rand(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.weights_hidden_output = np.random.rand(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.bias_output = np.zeros((1, output_dim))
        self.bias_hidden = np.zeros((1, hidden_dim))

    def predict(self, input_vector):
        hidden_input = np.dot(input_vector, self.weights_input_hidden) + self.bias_hidden
        hidden_output = np.tanh(hidden_input)
        output_layer_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output_layer_output = _softmax(output_layer_input)

        return np.argmax(output_layer_output, axis=1)

    def calculate_loss(self, input_vector, target_vector):
        hidden_input = np.dot(input_vector, self.weights_input_hidden) + self.bias_hidden
        hidden_output = np.tanh(hidden_input)
        output_layer_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output_layer_output = _softmax(output_layer_input)

        data_loss = -np.log(output_layer_output[range(input_vector.shape[0]), target_vector])
        return 1. / input_vector.shape[0] * np.sum(data_loss)

    def train(self, input_vector, target_vector):
        hidden_input = np.dot(input_vector, self.weights_input_hidden) + self.bias_hidden  # n x 3
        hidden_output = np.tanh(hidden_input)  # n x 3
        output_layer_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output  # n x 2
        output_layer_output = _softmax(output_layer_input)  # n x 2

        delta_hidden_output = np.copy(output_layer_output)
        delta_hidden_output[range(output_layer_output.shape[0]), target_vector] -= 1
        delta_weights_hidden_output = np.dot(np.transpose(hidden_output), delta_hidden_output)
        delta_bias_output = np.sum(delta_hidden_output, axis=0, keepdims=True)

        delta_input_hidden = np.dot(delta_hidden_output, np.transpose(self.weights_hidden_output)) \
            * (1 - np.power(np.tanh(hidden_input), 2))
        delta_weights_input_hidden = np.dot(np.transpose(input_vector), delta_input_hidden)
        delta_bias_hidden = np.sum(delta_input_hidden, axis=0)

        self.weights_hidden_output += -self.learning_rate * delta_weights_hidden_output
        self.weights_input_hidden += -self.learning_rate * delta_weights_input_hidden
        self.bias_output += -self.learning_rate * delta_bias_output
        self.bias_hidden += -self.learning_rate * delta_bias_hidden


def _softmax(input_vector):
    exp_scores = np.exp(input_vector)

    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
