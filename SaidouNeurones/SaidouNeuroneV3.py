import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class NeuralNetwork:
    def __init__(
        self,
        hidden_layers=(32, 32, 32),
        activation="sigmoid",
        learning_rate=0.1,
        regularization=None,
        reg_lambda=0.01,
    ):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.parameters = {}
        self.training_history = []

        if self.activation not in ['sigmoid', 'relu', 'tanh', 'softmax']:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def initialize_parameters(self, input_dim: int, output_dim: int):
        np.random.seed(1)
        layer_dims = [input_dim] + list(self.hidden_layers) + [output_dim]

        for l in range(1, len(layer_dims)):
            self.parameters[f'W{l}'] = (
                np.random.randn(layer_dims[l], layer_dims[l - 1]) *
                np.sqrt(2. / layer_dims[l - 1])
            )
            self.parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))

    def activate(self, Z: np.ndarray) -> np.ndarray:
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-Z))
        elif self.activation == "relu":
            return np.maximum(0, Z)
        elif self.activation == "tanh":
            return np.tanh(Z)
        elif self.activation == "softmax":
            expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            return expZ / np.sum(expZ, axis=0, keepdims=True)

    def derivative_activation(self, A: np.ndarray) -> np.ndarray:
        if self.activation == "sigmoid":
            return A * (1 - A)
        elif self.activation == "relu":
            return (A > 0).astype(float)
        elif self.activation == "tanh":
            return 1 - np.square(A)
        elif self.activation == "softmax":
            return 1  # not used directly

    def forward_propagation(self, X: np.ndarray) -> dict:
        activations = {'A0': X}
        L = len(self.hidden_layers) + 1
        for l in range(1, L + 1):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = W.dot(activations[f'A{l-1}']) + b
            activations[f'A{l}'] = self.activate(Z)
        return activations

    def compute_loss(self, A: np.ndarray, y: np.ndarray) -> float:
        m = y.shape[1]
        epsilon = 1e-15

        if self.activation == "softmax":
            loss = -np.mean(np.sum(y * np.log(A + epsilon), axis=0))
        elif self.activation == "sigmoid":
            loss = -np.mean(y * np.log(A + epsilon) + (1 - y) * np.log(1 - A + epsilon))
        else:
            raise ValueError("Unsupported activation for loss computation? Use 'sigmoid' or 'softmax'.")

        if self.regularization == "l2":
            l2_penalty = sum(np.sum(np.square(self.parameters[f'W{l}']))
                            for l in range(1, len(self.hidden_layers) + 2))
            loss += (self.reg_lambda / (2 * m)) * l2_penalty

        return loss

    def backward_propagation(self, X: np.ndarray, y: np.ndarray, activations: dict) -> dict:
        m = y.shape[1]
        L = len(self.hidden_layers) + 1
        grads = {}

        dZ = activations[f'A{L}'] - y  # Cross-entropy loss gradient

        for l in reversed(range(1, L + 1)):
            A_prev = activations[f'A{l-1}']
            grads[f'dW{l}'] = (1 / m) * dZ.dot(A_prev.T)
            grads[f'db{l}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if self.regularization == "l2":
                grads[f'dW{l}'] += (self.reg_lambda / m) * self.parameters[f'W{l}']

            if l > 1:
                dA_prev = self.parameters[f'W{l}'].T.dot(dZ)
                dZ = dA_prev * self.derivative_activation(activations[f'A{l-1}'])

        return grads

    def update_parameters(self, grads: dict):
        L = len(self.hidden_layers) + 1
        for l in range(1, L + 1):
            self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}']

    def predict(self, X: np.ndarray) -> np.ndarray:
        output = self.forward_propagation(X)[f'A{len(self.hidden_layers) + 1}']
        if self.activation == "softmax":
            return np.argmax(output, axis=0)
        else:
            return (output >= 0.5).astype(int)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_iterations: int = 1000,
        batch_size: int = None,
        validation_data: tuple = None,
        early_stopping_patience: int = None,
    ):
        self.initialize_parameters(X.shape[0], y.shape[0])
        m = X.shape[1]
        batch_size = batch_size or m
        best_val_loss = float("inf")
        patience_counter = 0

        history = []

        for i in tqdm(range(num_iterations)):
            idx = np.random.permutation(m)
            X_batch = X[:, idx[:batch_size]]
            y_batch = y[:, idx[:batch_size]]

            activations = self.forward_propagation(X_batch)
            grads = self.backward_propagation(X_batch, y_batch, activations)
            self.update_parameters(grads)

            A_train = activations[f'A{len(self.hidden_layers)+1}']
            train_loss = self.compute_loss(A_train, y_batch)
            val_loss = None

            if validation_data:
                X_val, y_val = validation_data
                A_val = self.forward_propagation(X_val)[f'A{len(self.hidden_layers)+1}']
                val_loss = self.compute_loss(A_val, y_val)

                if early_stopping_patience:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"Early stopping at iteration {i}")
                            break

            history.append((train_loss, val_loss))

        self.training_history = np.array(history)

        self.plot_losses()

    def plot_losses(self):
        if self.training_history is None:
            return

        plt.figure(figsize=(12, 4))
        plt.plot(self.training_history[:, 0], label='Train Loss')

        if self.training_history.shape[1] > 1:
            try:
                val_loss_column = self.training_history[:, 1].astype(float)
                if not np.all(np.isnan(val_loss_column)):
                    plt.plot(val_loss_column, label='Validation Loss')
            except (ValueError, TypeError):
                pass

        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        if self.activation == "softmax":
            return accuracy_score(np.argmax(y, axis=0), y_pred)
        else:
            return accuracy_score(y.flatten(), y_pred.flatten())
