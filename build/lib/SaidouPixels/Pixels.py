# Pixels.py

from . import Pixels
import numpy as np
import time

import matplotlib.pyplot as plt
import cv2


def Pixel(L: int, H: int, couche: int):
    ''' 
    Générent des 3 couches de pixels d'une valeurs aléatoire entre 0 et 255
    dtype = 'uint8'

    Arguments:
        L (int): Taille de largeur
        H (int): Taille de la hauteur
        couche (int): Nombre de couches par exemple : RGB (3 Couches)

    Returns:
        tuple = Un tuple contenant un tableau Numpy de taille L x H pixels et N(couche)
    '''
    pixelVal = np.random.randint(0, 256, size=(H, L, couche), dtype='uint8')
    return pixelVal, couche


def NormalizedTaille(image):
    plt.imshow(image)
    imgF = image.astype(float)
    imgNormalized = imgF / imgF.max()
    return imgNormalized


def NombrePixel(nbPixel: int):
    ''' 
    Cette fonction génère un tableau Numpy d'une ligne avec N pixels alignés.
    Les pixels sont initialisés avec des valeurs aléatoires entre 0 et 255.

    Arguments:
        nbPixel (int): Nombre de pixels

    Returns:
        pixels (ndArray): Tableau numpy contenant les pixels alignés.
    '''

    taille = nbPixel
    pixels = np.empty((taille, 1))

    for i in range(taille):
        pixels[i][0] = Pixel(np.random.randint(0, 255))

    return pixels


def GenerateImage(L: int, H: int, couche: int = 3, *, count: int = 1):
    """
    Génère et enregistre des images en utilisant la fonction Pixel.

    Arguments:
        L (int): Taille de largeur de l'image.
        H (int): Taille de hauteur de l'image.
        couche (int, optional): Nombre de couches de pixels. Par défaut, 3 (RGB).
        count (int, optional): Nombre d'images à générer. Par défaut, 1.

    Returns:
        None
    """

    for i in range(count):
        img = Pixel(L, H, couche)
        nFile = f"./DataImage/img{i}.jpg"
        cv2.imwrite(nFile, img)


def ChargerImage(img: str, count: int = 1):
    """
    Charge les images à partir du répertoire spécifié et les stocke dans un tableau numpy.

    Arguments:
        img (str): Le nom de base des fichiers d'images.
        count (int, optional): Le nombre d'images à charger. Par défaut, 1.
        resize (tuple, optional): La taille de redimensionnement des images au format (largeur, hauteur). Par défaut, None.

    Returns:
        tuple: Un tuple contenant le tableau numpy des images chargées et le tableau numpy du nombre d'images chargées.
    """
    dataX = []
    dataY = []
    for i in range(count):
        nFile = f"./DataImage/{img}{i}.jpg"
        image = plt.imread(nFile)
        dataX.append((image))
        dataY.append(count)
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    return dataX, dataY


def decompte(interval: float, temps: int):
    ''' 
    Cette fonction crée un décompte avec un temps d'intervalle
    qui permet d'exécuter une tâche

    Arguments:
        interval (float): Le temps d'intervalle entre chaque décompte
        temps (int): Le temps du décompte

    Returns:
        None
    '''
    t = temps
    for i in range(t):
        print("Décompte.. {i} seconde..")
        t -= time.time()
        time.sleep(interval)
        print("Fini !")
