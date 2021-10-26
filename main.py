import sklearn
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
from scipy.stats import shapiro
import seaborn as sns
#%matplotlib notebook


### APARTAT (C): ANALITZANT DADES

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


# Apliquem el test de Shapiro per veure si es segueix una distribució Gaussiana
columns = dataset.columns
for column in columns:
    normal, value = (shapiro(dataset[column]))
    rounded_value = round(value, 5)
    if rounded_value > 0.05:
        print('Probably Gaussian')
    else:
        print("Probably not Gaussian")
# No tenim cap columna del dataset que seguieixi una distribució Gaussiana


# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
correlacio = dataset.corr()

plt.figure()
plt.figure(figsize=(16, 5))
ax = sns.heatmap(correlacio, annot=True, linewidths=.5)

#en el dataset se habla de la correlación entre Rmag y mumax para sacar los tamaños de las galaxias
#si bien es cierto que nos encontramos que cierto grupo de atributos son muc correlativos
#(el cuadrado ese que se ve, decir las columnas), hay que tener en cuenta que son datos muy parecidos/relacionados (bandas
#cercanas en la medición), por lo que no se saca mucho de ahí

#comprobamos la correlación entre estos dos datos mediante gráficos

relacio = sns.pairplot(dataset[["Rmag", "ApDRmag", "Mcz", "mumax"]])

rel_Rmag_mumax = dataset[["Rmag","mumax"]]
relacio = sns.pairplot(rel_Rmag_mumax)
plt.show()


#Si bien la correlación es buena, es necesario normalizar los datos para hacerlos más consistentes
#suprimimos los outliers con z_score, que nos indica lo lejos que está cada dato de la media(?) del atributo
#(https://www.statisticshowto.com/probability-and-statistics/z-score/)

from scipy import stats
rel_Rmag_mumax2 = rel_Rmag_mumax[(np.abs(stats.zscore(rel_Rmag_mumax)) < 2).all(axis=1)]

#comprobamos la correlación entre estos dos datos mediante gráficos
relacio = sns.pairplot(rel_Rmag_mumax2)
plt.show()

"""columnsx = rel_Rmag_mumax2.columns
for column in columnsx:
    normal, value = (shapiro(dataset[column]))
    rounded_value = round(value, 10)
    if rounded_value > 0.05:
        print('Probably Gaussian')
    else:
        print("Probably not Gaussian")
    print(rounded_value)
    print(value)"""

#Vemos que siguen sin ser gaussianas pero al menos hemos quitado parte de los outliers
# aún podemos ajustar el dataset con Chi-square y su relación con los outliers + indicaciones del propio dataset que
#indican qué datos descartar (galaxias débiles, con mucho error en algún atributo como para ser fiables, etc)
#probemos la efectividad de hacer predicciones con este dataset


### APARTAT (B): PRIMERES REGRESSIONS