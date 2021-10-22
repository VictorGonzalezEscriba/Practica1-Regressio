import sklearn
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
from scipy.stats import shapiro
#%matplotlib notebook

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
listt = lst[-10:]

# Quitamos mumax y ApDRmag
listt.append('mumax')
listt.append('ApDRmag')

# Eliminem valors inexistents
dataset = dataset.dropna()

# Drop de las columnas que den info acerca de los errores
for x in lst:
    if "e." in x:
        listt.append(x)
dataset = dataset.drop(columns=listt)
# print(dataset)
print("Total de valors no existents:", dataset.isnull().sum().sum())

# Apliquem el test de Shapiro
columns = dataset.columns
for column in columns:
    normal, value = (shapiro(dataset[column]))
    rounded_value = round(value, 5)
    if rounded_value > 0.05:
        print('Probably Gaussian')
    else:
        print("Probably not Gaussian")
# No tenim cap columna del dataset que seguieixi una distribució Gaussiana

import seaborn as sns

# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
correlacio = dataset.corr()

plt.figure()
# Para verlo más grande
plt.figure(figsize=(16, 5))
ax = sns.heatmap(correlacio, annot=True, linewidths=.5)
pd.plotting.scatter_matrix(dataset, alpha=0.2)
