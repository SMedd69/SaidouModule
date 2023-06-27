import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm

''' ---- Exemple : ----
         
        /-\ 
       / | \
      /  |  \
     /   |   \   ====> import LoadData.py <====
    /    o    \
   /___________\
      
    # Chat vs Chien
    # X_train, y_train, X_test, y_test = Load_data(
    #     './Deep-Learning-Youtube-main/datasets/trainset.hdf5', './Deep-Learning-Youtube-main/datasets/testset.hdf5')

    # y_train = y_train.T
    # y_test = y_test.T

    # X_train = X_train.T
    # X_train_Reshape = X_train.reshape(-1, X_train.shape[-1]) / X_train.max()

    # X_test = X_test.T
    # X_test_Reshape = X_test.reshape(-1, X_test.shape[-1]) / X_train.max()

    # m_train = 64
    # m_test = 64
    # X_test_Reshape = X_test_Reshape[:, :m_test]
    # X_train_Reshape = X_train_Reshape[:, :m_train]
    # y_train = y_train[:, :m_train]
    # y_test = y_test[:, :m_test]

 '''

# 1 - Initialisations


def Initialisation(Dimensions: dict):
    ''' 
    Fonction d'initialisation des paramètres W, b qui
    permet d'initialisé les poids et les dimensions
    qui retourne un dictionnaire contenant les poids

    Attributs:
    Dimensions : La fonction d'initialisation
                 prend en paramètre un dictionnaire

    returns :
    Parametres(dict) : retourne un dictionnaire 
                       de valeurs des poids W, b initialisés
    '''
    Parametres = {}
    C = len(Dimensions)

    for c in range(1, C):
        Parametres['W' +
                   str(c)] = np.random.randn(Dimensions[c], Dimensions[c - 1])
        Parametres['b' + str(c)] = np.random.randn(Dimensions[c], 1)

    return Parametres

# 2 - Propagation en avant


def Forward_Propagation(X, Parametres):
    ''' 
    La propagation en avant permet d'activé les neurones
    Cette fonction prend en paramètre l'entrée X et les 
    paramètres W, b et retourne un dictionnaire Activations.
    L'activation se fait avec le sigma et produit une sortie
    qui prend des valeurs entre [0, 1]

    Attributs :
    X : Entrée des données d'entraînement
    Paramètres (dict): Dictionnaire des paramètres W, b
                       initialisés plutôt avec la fonction
                       d'initialisation

    Returns :
    Activations (dict): Dictionnaire contenant les poids
    activés

    '''
    Activations = {'A0': X}
    C = len(Parametres) // 2
    for c in range(1, C + 1):
        Z = Parametres['W' + str(c)].dot(Activations['A' +
                                                     str(c-1)]) + Parametres['b' + str(c)]
        Activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
    return Activations

#  3 - Fonction coût - Calcul de pertes


def Log_Loss(A, y):
    ''' 
    Fonction coût permet de calculer l'erreur entre
    la valeur produite et la valeur attendue
    en ajoutant un epsilon = 1e-15

    Attributs :
    A (dict) : Dictionnaires des Activations des poids
    y : Valeur attendue

    Returns :

    La sommes de la différence en logarithme de la valeurs
    produite moins la valeur attendue
    '''
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

# 4 - Propagation en arrière


def Back_Propagation(y, Activations, Parametres):
    ''' 
    Propagation en arrière
    '''
    m = y.shape[1]
    C = len(Parametres) // 2

    dZ = Activations['A' + str(C)] - y
    Gradients = {}

    for c in reversed(range(1, C + 1)):
        Gradients['dW' + str(c)] = 1 / m * np.dot(dZ,
                                                  Activations['A' + str(c - 1)].T)
        Gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        if c > 1:
            dZ = np.dot(Parametres['W' + str(c)].T, dZ) * Activations['A' +
                                                                      str(c - 1)] * (1 - Activations['A' + str(c - 1)])

    return Gradients

# 5 - Fonction de mise à jour à près la propagation en arrières et ajustements des poids et des biais


def Update(Gradients, Parametres, learning_rate):
    ''' 
    Mise à jour des poids après minimisation
    '''
    C = len(Parametres) // 2

    for c in range(1, C + 1):
        Parametres['W' + str(c)] = Parametres['W' + str(c)] - \
            learning_rate * Gradients['dW' + str(c)]
        Parametres['b' + str(c)] = Parametres['b' + str(c)] - \
            learning_rate * Gradients['db' + str(c)]
    return Parametres

# 6 - Prediction


def Predict(X, Parametres):
    ''' 

    La fonction Predict permet de tester le réseau 
    sur des données inconnus.

    Retourne un dictionnaire si la valeur de la prédiction
    est >= à 0.5

    '''
    Activations = Forward_Propagation(X, Parametres)
    C = len(Parametres) // 2
    AF = Activations['A' + str(C)]

    return AF >= 0.5


def Artificial_neurone(X, y, hidden_layers=(32, 32, 32), learning_rate=0.1, n_iter=1000):
    ''' 
    Fonction regroupe le réseau en choisissant la taille des couches
    retourne en dictionnaire contenant les paramètres et le modèle
    entraîner.
    '''

    # initialiser
    Dimensions = list(hidden_layers)
    Dimensions.insert(0, X.shape[0])
    Dimensions.append(y.shape[0])
    np.random.seed(1)
    Parametres = Initialisation(Dimensions)

    training_history = np.zeros((int(n_iter), 2))

    C = len(Parametres) // 2

    for i in tqdm(range(n_iter)):

        Activations = Forward_Propagation(X, Parametres)
        Gradients = Back_Propagation(y, Activations, Parametres)
        Parametres = Update(Gradients, Parametres, learning_rate)
        Af = Activations['A' + str(C)]

        training_history[i, 0] = (Log_Loss(y.flatten(), Af.flatten()))
        y_pred = Predict(X, Parametres)
        print(y_pred)
        training_history[i, 1] = (
            accuracy_score(y.flatten(), y_pred.flatten()))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='train Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label='train acc')
    plt.legend()
    plt.show()

    # Paramètres d'entrainement à sauvegarder !!
    return Parametres

# parametres1 = Artificial_neurone(X_train_Reshape, y_train, hidden_layers=(
#     64, 32, 16), learning_rate=0.2, n_iter=900)
