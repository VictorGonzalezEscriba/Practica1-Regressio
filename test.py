import sklearn
import numpy as np
import matplotlib
import scipy
import pandas as pd
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import scipy.stats
import seaborn as sns

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


dataset1 = dataset
dataset1.describe()

dataTypeDict = dict(dataset1.dtypes)

#creamos una lista para seleccionar los nombres de columnas que queremos dropear
lst = []

print(dataTypeDict)

for x in dataTypeDict:
    lst.append(x)
    # print(x)
    # print(dataTypeDict[x])


# drop de la columna de indice
dataset1 = dataset1.drop(columns='Nr')

# quitamos las columnas entre la 56 y la 65 al ser valores redundantes de las 13 anteriores
listt = lst[-10:]

# quitamos mumax y ApDRmag
listt.append('mumax')
listt.append('ApDRmag')

# is_numeric_dtype(df['B'])

# drop de las columnas que den info acerca de los errores
for x in lst:
    if "e." in x:
        listt.append(x)

dataset1 = dataset1.drop(columns=listt)
print(dataset1)


# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
correlacio = dataset1.corr()

plt.figure()

ax = sns.heatmap(correlacio, annot=True, linewidths=.1)

print(dataset1["VjMAG"].describe())