import sklearn
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
from scipy.stats import shapiro

from sklearn.model_selection import train_test_split
import numpy as np #importem la llibreria


# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset
# Carreguem dataset d'exemple
dataset = load_dataset('COMBO17.csv')
data = dataset.values

x = data[:, :2]
y = data[:, 2]

# Per veure les dimensionalitats
print("Dimensionalitat de la BBDD:", dataset.shape)
print("Dimensionalitat de les entrades X:", x.shape)
print("Dimensionalitat de l'atribut Y:", y.shape)


# Per veure totes les columnes amb valors inexistents
x = dataset.isnull().sum()
null_columns = dataset.columns[dataset.isnull().any()]
print(null_columns)
print(dataset[null_columns].isnull().sum())
print("Total de valors no existents:",dataset.isnull().sum().sum())
# Tipus de cada atribut:
print(dataset.dtypes)


# Creamos una lista para seleccionar los nombres de columnas que queremos dropear
dataTypeDict = dict(dataset.dtypes)
lst = []
for x in dataTypeDict:
    lst.append(x)

# Drop de la columna 1 que contiene el indice
dataset = dataset.drop(columns='Nr')

# Quitamos las columnas entre la 56 y la 65 al ser valores redundantes de las 13 anteriores
lst2 = lst[-10:]
# Eliminem valors inexistents
dataset = dataset.dropna()

# Drop de las columnas que den info acerca de los errores
for x in lst:
    if "e." in x:
        lst2.append(x)
dataset = dataset.drop(columns=lst2)
print("Total de valors no existents:", dataset.isnull().sum().sum())