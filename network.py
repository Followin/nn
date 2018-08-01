import numpy as np


# input: 1 x 2
# w_i_h 2 x 3
# w_h_o 3 x 2

class NeuralNetwork:
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim, learning_rate, regularization_lambda):
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.regularization_lambda = regularization_lambda

        self.hidden1_activation_func = lambda input_vector: 1 / (1 + np.exp(-input_vector))
        self.hidden1_activation_func_der = lambda input_vector: self.hidden1_activation_func(input_vector) \
            * (1 - self.hidden1_activation_func(input_vector))

        self.hidden2_activation_func = np.tanh
        self.hidden2_activation_func_der = lambda input_vector: 1 - np.power(np.tanh(input_vector), 2)

        # self.hidden_activation_func = np.arctan
        # self.hidden_activation_func_der = lambda input_vector: 1 / (np.power(input_vector, 2) + 1)

        np.random.seed(0)
        self.weights_input_hidden1 = np.random.rand(input_dim, hidden1_dim) / np.sqrt(input_dim)
        self.weights_hidden1_hidden2 = np.random.rand(hidden1_dim, hidden2_dim) / np.sqrt(hidden1_dim)
        self.weights_hidden2_output = np.random.rand(hidden2_dim, output_dim) / np.sqrt(hidden2_dim)
        self.bias_output = np.zeros((1, output_dim))
        self.bias_hidden1 = np.zeros((1, hidden1_dim))
        self.bias_hidden2 = np.zeros((1, hidden2_dim))

    def predict(self, input_vector):
        hidden1_input = np.dot(input_vector, self.weights_input_hidden1) + self.bias_hidden1
        hidden1_output = self.hidden1_activation_func(hidden1_input)
        hidden2_input = np.dot(hidden1_output, self.weights_hidden1_hidden2) + self.bias_hidden2
        hidden2_output = self.hidden2_activation_func(hidden2_input)
        output_layer_input = np.dot(hidden2_output, self.weights_hidden2_output) + self.bias_output
        output_layer_output = _softmax(output_layer_input)

        return np.argmax(output_layer_output, axis=1)

    def calculate_loss(self, input_vector, target_vector):
        hidden1_input = np.dot(input_vector, self.weights_input_hidden1) + self.bias_hidden1
        hidden1_output = self.hidden1_activation_func(hidden1_input)
        hidden2_input = np.dot(hidden1_output, self.weights_hidden1_hidden2) + self.bias_hidden2
        hidden2_output = self.hidden2_activation_func(hidden2_input)
        output_layer_input = np.dot(hidden2_output, self.weights_hidden2_output) + self.bias_output
        output_layer_output = _softmax(output_layer_input)

        data_loss = -np.log(output_layer_output[range(input_vector.shape[0]), target_vector])
        return 1. / input_vector.shape[0] * np.sum(data_loss)

    def train(self, input_vector, target_vector):
        hidden1_input = np.dot(input_vector, self.weights_input_hidden1) + self.bias_hidden1
        hidden1_output = self.hidden1_activation_func(hidden1_input)
        hidden2_input = np.dot(hidden1_output, self.weights_hidden1_hidden2) + self.bias_hidden2
        hidden2_output = self.hidden2_activation_func(hidden2_input)
        output_layer_input = np.dot(hidden2_output, self.weights_hidden2_output) + self.bias_output
        output_layer_output = _softmax(output_layer_input)

        delta_hidden2_output = np.copy(output_layer_output)
        delta_hidden2_output[range(output_layer_output.shape[0]), target_vector] -= 1
        delta_weights_hidden2_output = np.dot(np.transpose(hidden2_output), delta_hidden2_output)
        delta_bias_output = np.sum(delta_hidden2_output, axis=0, keepdims=True)

        delta_hidden1_hidden2 = np.dot(delta_hidden2_output, np.transpose(self.weights_hidden2_output)) \
            * self.hidden2_activation_func_der(hidden2_input)
        delta_weights_hidden1_hidden2 = np.dot(np.transpose(hidden1_output), delta_hidden1_hidden2)
        delta_bias_hidden2 = np.sum(delta_hidden1_hidden2, axis=0)

        delta_input_hidden1 = np.dot(delta_hidden1_hidden2, np.transpose(self.weights_hidden1_hidden2)) \
            * self.hidden1_activation_func_der(hidden1_input)
        delta_weights_input_hidden1 = np.dot(np.transpose(input_vector), delta_input_hidden1)
        delta_bias_hidden1 = np.sum(delta_input_hidden1, axis=0)

        self.weights_hidden2_output += -self.learning_rate * (delta_weights_hidden2_output + self.regularization_lambda * self.weights_hidden2_output)
        self.weights_hidden1_hidden2 += -self.learning_rate * (delta_weights_hidden1_hidden2 + self.regularization_lambda * self.weights_hidden1_hidden2)
        self.weights_input_hidden1 += -self.learning_rate * (delta_weights_input_hidden1 + self.regularization_lambda * self.weights_input_hidden1)
        self.bias_output += -self.learning_rate * delta_bias_output
        self.bias_hidden2 += -self.learning_rate * delta_bias_hidden2
        self.bias_hidden1 += -self.learning_rate * delta_bias_hidden1


def _softmax(input_vector):
    exp_scores = np.exp(input_vector)

    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
