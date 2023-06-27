from setuptools import setup

setup(
    name='SaidouModule',
    version='1.0.1',
    author='Meddahi Saïd',
    description='Module possédant 2 packages : SaidouPixels et SaidouNeurone, lire le README.md pour plus d\'informations.',
    packages=['SaidouPixels', 'SaidouNeurones'],
    install_requires=['numpy',
                      'matplotlib',
                      'opencv-python',
                      'tqdm',
                      'scikit-learn'
                      ])
