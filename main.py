import sklearn
import numpy as np
import matplotlib
import scipy
import pandas as pd
from scipy.stats import norm
from scipy.stats import shapiro
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import scipy.stats

# Apartat (C): Analitzant Dades

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


print("Per comptar el nombre de valors no existents:")
print(dataset.isnull().sum().sum())
# Para ver todas las columnas con valores vacios

x = dataset.isnull().sum()
null_columns = dataset.columns[dataset.isnull().any()]
print(null_columns)
print(dataset[null_columns].isnull().sum())

# VnMag 1, e.VbMAG 1, S280MAG 24, e.S280MA 24


null_columns = dataset.columns[dataset.isnull().any()]

for e in null_columns:
    mean_value = dataset[e].mean()
    dataset[e].fillna(value = mean_value, inplace=True)

x = dataset.isnull().sum()
null_columns = dataset.columns[dataset.isnull().any()]
print(null_columns)
print(dataset[null_columns].isnull().sum())

print(dataset.isnull().sum().sum())


ee = dataset.columns

print(type(ee))
print(shapiro(dataset[ee[0]]))
dataset.dropna()
dataset.drop(columns='Nr')
print(dataset.isnull().sum().sum())


i = 0
while i < ee.size:
    normal = (shapiro(dataset[ee[i]]))
    i += 1
