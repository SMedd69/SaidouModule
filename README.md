# [SaidouModule](https://pypi.org/project/SaidouModule/)

Le module [**SaidouModule**](https://pypi.org/project/SaidouModule/) contient deux répertoires : [**SaidouNeurones**](SaidouNeurones) et [**SaidouPixels**](SaidouPixels).
- Pour installer le module : 
```shell
pip install SaidouModule
```

Voici une description des scripts présents dans chaque répertoire :

## [SaidouNeurones](SaidouNeurones)

- [**ReseauNeuronesSaid.py**](SaidouNeurones/ReseauNeuronesSaid.py)
  Ce script implémente un réseau de neurones artificiels.
  Exemple d'utilisation :

  ```python
  import SaidouNeurones.ReseauNeuronesSaid as rns

  # Exemple d'utilisation du réseau de neurones
  X = rns.np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
  y = rns.np.array([[0, 1, 1, 0]])
  model = rns.Artificial_neurone(X, y, hidden_layers=(2,), learning_rate=0.1, n_iter=1000)
  predictions = rns.model.predict(X)
  print(predictions)
  ```

- **SaidouNeuronesV2.py**
  Ce script implémente une classe NeuralNetwork pour un réseau de neurones.
  Exemple d'utilisation :

  ```python
  import SaidouNeurones.SaidouNeuronesV2 sn2

  # Exemple d'utilisation du réseau de neurones
  X = sn2.np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
  y = sn2.np.array([[0, 1, 1, 0]])
  model = sn2.NeuralNetwork(hidden_layers=(2,), activation='sigmoid', learning_rate=0.1)
  model.fit(X, y, num_iterations=1000)
  predictions = model.predict(X)
  print(predictions)
  ```

## [SaidouPixels](SaidouPixels)

- [**Pixels**](SaidouPixels/Pixels.py)
  Ce script implémente des fonctionalités dédier aux images qui stock les valeurs des pixels dans une matrice **Numpy** de dimension *(Largeur, Hauteur, Nombre de couches)* pour une image et *(Nombres d'images, Largeur des images, Hauteur des images, Couches (ex = 3 pour RGB))* pour un lot d'images.

  Le script [**Pixels**](SaidouPixels/Pixels.py) possède aussi des fonctionalitès permettant de générer des images en choisissant la taille, la couche (par ex: 2 pour une image en noir et blanc) ainsi que le nombre d'images et charger des images en choisissant également le nombre d'image à charger et d'autres fonctionnalités dédier au traitements d'images.

  - Les valeurs des pixels sont données aléatoirement entre 0 et 255 (int)

  Exemple d'utilisation :

  ```python
  import SaidouPixels as sp

  # Exemple d'utilisation de la fonction Pixel
  val_pixels = sp.Pixel(L=28, H=28, couches=3)
  print(val_pixels)
  
  # Affichage de l'image
  sp.plt.imshow(val_pixels)
  ```

## À propos

Ce module [**SaidouModule**](https://pypi.org/project/SaidouModule/) a été créé par mes soins dans un but purement éducatif et pédagogique dans un premier temps, puis en fonction des évolutions du projets ce module aura pour but d'implémenter des fonctionalités necessaire aux réseau de neurones artificielle. 

Interessé par se projet ? Ne pas hésiter à me contacter à mon adresse mail suivante  : <a href="mailto:s.meddahi.fo@gmail.com">s.meddahi.fo@gmail.com</a>

## License

Ce projet est sous licence MIT.
