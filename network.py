import numpy as np


# input: 1 x 2
# w_i_h 2 x 3
# w_h_o 3 x 2

# { number: int, activation_func: lambda, activation_func_der: lambda }

class NeuralNetwork:
    def __init__(self, input_dim, hidden_layers, output_dim, learning_rate, regularization_lambda):
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.regularization_lambda = regularization_lambda

        np.random.seed(0)

        self.weights_hidden = []

        weights_input_hiddenhead = np.random.rand(input_dim, hidden_layers[0]['dimension']) / np.sqrt(input_dim)
        self.weights_hidden.append(weights_input_hiddenhead)

        for i, current_layer in list(enumerate(hidden_layers))[1:]:
            previous_layer = hidden_layers[i - 1]
            weights = np.random.rand(previous_layer['dimension'], current_layer['dimension']) / np.sqrt(previous_layer['dimension'])
            self.weights_hidden.append(weights)

        weights_hiddentail_output = np.random.rand(hidden_layers[-1]['dimension'], output_dim) / np.sqrt(hidden_layers[-1]['dimension'])
        self.weights_hidden.append(weights_hiddentail_output)

        self.biases_hidden = []
        for layer in hidden_layers:
            self.biases_hidden.append(np.zeros((1, layer['dimension'])))
        self.biases_hidden.append(np.zeros((1, output_dim)))

    def predict(self, input_vector):
        outputs = [input_vector]
        for i, layer in enumerate(self.hidden_layers):
            layer_input = np.dot(outputs[-1], self.weights_hidden[i]) + self.biases_hidden[i]
            outputs.append(layer['activation_func'](layer_input))

        output_layer_input = np.dot(outputs[-1], self.weights_hidden[-1]) + self.biases_hidden[-1]
        output_layer_output = _softmax(output_layer_input)

        return np.argmax(output_layer_output, axis=1)

    def calculate_loss(self, input_vector, target_vector):
        outputs = [input_vector]
        for i, layer in enumerate(self.hidden_layers):
            layer_input = np.dot(outputs[-1], self.weights_hidden[i]) + self.biases_hidden[i]
            outputs.append(layer['activation_func'](layer_input))

        output_layer_input = np.dot(outputs[-1], self.weights_hidden[-1]) + self.biases_hidden[-1]
        output_layer_output = _softmax(output_layer_input)

        data_loss = -np.log(output_layer_output[range(input_vector.shape[0]), target_vector])
        return 1. / input_vector.shape[0] * np.sum(data_loss)

    def train(self, input_vector, target_vector):
        outputs = [input_vector]
        inputs = [input_vector]

        for i, layer in enumerate(self.hidden_layers):
            layer_input = np.dot(outputs[-1], self.weights_hidden[i]) + self.biases_hidden[i]
            inputs.append(layer_input)
            outputs.append(layer['activation_func'](layer_input))

        output_layer_input = np.dot(outputs[-1], self.weights_hidden[-1]) + self.biases_hidden[-1]
        output_layer_output = _softmax(output_layer_input)

        delta_hiddentail_output = np.copy(output_layer_output)
        delta_hiddentail_output[range(output_layer_output.shape[0]), target_vector] -= 1
        delta_weights_hiddentail_output = np.dot(np.transpose(outputs[-1]), delta_hiddentail_output)

        weights_deltas = [delta_weights_hiddentail_output]
        bias_deltas = [np.sum(delta_hiddentail_output, axis=0, keepdims=True)]

        layer_deltas = [delta_hiddentail_output]
        for i, layer in reversed(list(enumerate(self.hidden_layers))):
            layer_delta = np.dot(layer_deltas[-1], np.transpose(self.weights_hidden[i + 1])) * layer['activation_func_der'](inputs[i + 1])
            weights_delta = np.dot(np.transpose(outputs[i]), layer_delta)
            bias_delta = np.sum(layer_delta, axis=0)

            layer_deltas.append(layer_delta)
            weights_deltas.insert(0, weights_delta)
            bias_deltas.insert(0, bias_delta)

        for i in range(len(self.weights_hidden)):
            self.weights_hidden[i] += -self.learning_rate * (weights_deltas[i] + self.regularization_lambda * self.weights_hidden[i])
            self.biases_hidden[i] += -self.learning_rate * bias_deltas[i]


def _softmax(input_vector):
    exp_scores = np.exp(input_vector)

    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
