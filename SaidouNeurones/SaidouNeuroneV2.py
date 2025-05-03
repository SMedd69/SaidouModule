import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from tqdm import tqdm


class NeuralNetwork:
    def __init__(self, hidden_layers=(32, 32, 32), activation='sigmoid', learning_rate=0.1, regularization=None,
                 reg_lambda=0.01):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.parameters = {}
        self.training_history = None

    def initialize_parameters(self, input_dim, output_dim):
        np.random.seed(1)
        dimensions = [input_dim] + list(self.hidden_layers) + [output_dim]
        num_layers = len(dimensions)
        for l in range(1, num_layers):
            self.parameters['W' + str(l)] = np.random.randn(dimensions[l], dimensions[l - 1]) * np.sqrt(
                2 / dimensions[l - 1])
            self.parameters['b' + str(l)] = np.zeros((dimensions[l], 1))

    def activate(self, Z):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif self.activation == 'relu':
            return np.maximum(0, Z)
        elif self.activation == 'tanh':
            return np.tanh(Z)
        elif self.activation == 'softmax':
            expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            return expZ / np.sum(expZ, axis=0, keepdims=True)

    def forward_propagation(self, X):
        activations = {'A0': X}
        num_layers = len(self.hidden_layers) + 1
        for l in range(1, num_layers + 1):
            Z = self.parameters['W' + str(l)].dot(
                activations['A' + str(l - 1)]) + self.parameters['b' + str(l)]
            activations['A' + str(l)] = self.activate(Z)
        return activations

    def compute_loss(self, A, y):
        epsilon = 1e-15
        m = y.shape[1]
        if self.activation == 'softmax':
            loss = -np.mean(y * np.log(A + epsilon))
        else:
            loss = -np.mean(y * np.log(A + epsilon) + (1 - y)
                            * np.log(1 - A + epsilon))
        if self.regularization == 'l2':
            l2_reg = 0
            num_layers = len(self.hidden_layers) + 1
            for l in range(1, num_layers + 1):
                l2_reg += np.sum(np.square(self.parameters['W' + str(l)]))
            loss += (self.reg_lambda / (2 * m)) * l2_reg
        return loss

    def backward_propagation(self, X, y, activations):
        m = y.shape[1]
        num_layers = len(self.hidden_layers) + 1
        grads = {}

        dZ = activations['A' + str(num_layers)] - y

        for l in reversed(range(1, num_layers + 1)):
            grads['dW' + str(l)] = (1 / m) * np.dot(dZ,
                                                    activations['A' + str(l - 1)].T)
            grads['db' + str(l)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if self.regularization == 'l2':
                grads['dW' + str(l)] += (self.reg_lambda / m) * \
                    self.parameters['W' + str(l)]

            if l > 1:
                dZ = np.dot(self.parameters['W' + str(l)].T, dZ) * \
                    self.derivative_activation(activations['A' + str(l - 1)])

        return grads

    def update_parameters(self, grads):
        num_layers = len(self.hidden_layers) + 1
        for l in range(1, num_layers + 1):
            self.parameters['W' +
                            str(l)] -= self.learning_rate * grads['dW' + str(l)]
            self.parameters['b' +
                            str(l)] -= self.learning_rate * grads['db' + str(l)]

    def derivative_activation(self, A):
        if self.activation == 'sigmoid':
            return A * (1 - A)
        elif self.activation == 'relu':
            return np.where(A <= 0, 0, 1)
        elif self.activation == 'tanh':
            return 1 - np.square(A)
        elif self.activation == 'softmax':
            return 1  # The derivative is not required for softmax activation

    def predict(self, X):
        activations = self.forward_propagation(X)
        output_activation = activations['A' + str(len(self.hidden_layers) + 1)]
        if self.activation == 'softmax':
            return np.argmax(output_activation, axis=0)
        else:
            return (output_activation >= 0.5).astype(int)

    def fit(self, X, y, num_iterations=1000, batch_size=None, validation_data=None, early_stopping_patience=None):
        self.initialize_parameters(X.shape[0], y.shape[0])
        self.training_history = np.zeros((num_iterations, 2))

        if batch_size is None:
            batch_size = X.shape[1]

        best_val_loss = float('inf')
        patience_count = 0

        for i in tqdm(range(num_iterations)):
            if batch_size < X.shape[1]:
                random_indices = np.random.choice(
                    X.shape[1], size=batch_size, replace=False)
                X_batch = X[:, random_indices]
                y_batch = y[:, random_indices]
            else:
                X_batch = X
                y_batch = y

            activations = self.forward_propagation(X_batch)
            grads = self.backward_propagation(X_batch, y_batch, activations)
            self.update_parameters(grads)

            train_loss = self.compute_loss(
                activations['A' + str(len(self.hidden_layers) + 1)], y_batch)
            self.training_history[i, 0] = train_loss

            if validation_data is not None:
                val_loss = self.compute_loss(self.predict(
                    validation_data[0]), validation_data[1])
                self.training_history[i, 1] = val_loss

                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_count = 0
                    else:
                        patience_count += 1

                    if patience_count >= early_stopping_patience:
                        print("Early stopping: Validation loss did not improve for {} iterations.".format(
                            early_stopping_patience))
                        break

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history[:, 0], label='Train Loss')
        if validation_data is not None:
            plt.plot(self.training_history[:, 1], label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.training_history[:, 1], label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(np.argmax(
            y, axis=0), y_pred) if self.activation == 'softmax' else accuracy_score(y, y_pred)
        return accuracy
