import numpy as np
from SaidouNeurones.SaidouNeuroneV3 import NeuralNetwork

def test_training_accuracy_on_linearly_separable_data():
    # Générer un dataset simple pour test (classification binaire)
    np.random.seed(0)
    X = np.random.randn(2, 200)
    y = (X[0, :] + X[1, :] > 0).astype(int).reshape(1, -1)

    # Initialiser et entraîner le modèle
    model = NeuralNetwork(hidden_layers=(10,), activation='sigmoid', learning_rate=0.1)
    model.fit(X, y, num_iterations=500)

    # Prédictions et précision
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)

    # Assertion
    assert accuracy > 0.9, f"L'exactitude attendue > 90 %, mais obtenue : {accuracy*100:.2f}%"
