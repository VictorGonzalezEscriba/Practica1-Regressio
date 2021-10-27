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

plt.figure(figsize=(16, 5))
ax = sns.heatmap(correlacio, annot=True, linewidths=.5)
plt.show()

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

# CÁLCULO DEL ERROR CUADRÁTICO

import math

def mean_squeared_error(y1, y2):
    # comprovem que y1 i y2 tenen la mateixa mida
    assert(len(y1) == len(y2))
    mse = np.sum((y1-y2)**2)
    return mse / len(y1)


### APARTAT (C): EL DESCENS DEL GRADIENT
class Regressor(object):
    def __init__(self, w0, w1, alpha, lamda): # si cambiamos w0 y w1 a menos hará más iteraciones
        # Inicialitzem w0 i w1 (per ser ampliat amb altres w's)
        self.w0 = w0
        self.w1 = w1
        self.alpha = alpha
        # added
        self.lamda = lamda
        # guardem jw per fer gràfiques
        self.cost = []
        # per indicar quantes iteracions fer
        self.n_iteracions = 0
        # m son las entradas del dataset = len(hy)

    def predict(self, x):
        # implementar aqui la funció de prediccio
        # f(x) = w1x + w0 -> retorna un vector amb totes les posiciones
        #print((self.w1 * x) + self.w0)
        return (self.w1 * x) + self.w0

    def __update(self, dy, x):
        # actualitzar aqui els pesos donada la prediccio (dy) i la y real.
        # cojo el dataset y calculo el error
        self.w0 = self.w0 - ((self.alpha/len(dy)) * np.sum(dy))
        m = (self.alpha * self.lamda / len(dy))
        self.w1 = self.w1 * (1 - m) - m * np.sum(dy * x)


    def train(self, max_iter, x, y):
        # Entrenar durant max_iter iteracions o fins que la millora sigui inferior a epsilon
        # x los datos con los que predecir
        # y datos target lo que tendria que dar
        jw_prev = 10000000
        for i in range(0, max_iter):
            hy = self.predict(x)
            diff_hy = hy - y
            jw = (1 / (2 * len(diff_hy))) * np.sum(diff_hy ** 2) + (self.lamda * ((self.w0 ** 2) + (self.w1 ** 2)))
            self.cost.append(jw) # hacer un plot que sea
            if (np.abs(jw - jw_prev) / jw_prev) < 0.05: # si lo hacemos más pequeño, hace más iteraciones, margen de error / porcentaje de mejora
                break
            self.__update(diff_hy, x)
            jw_prev = jw


"""
    r = Regressor(-10, -10, 0.05, 0.05)
    data = rel_Rmag_mumax.to_numpy()
    x = data[:, 0]
    y = data[:, 1]
    r.train(100, x, y)
"""


"""
antes alpha y lamba nos hacia avanzar muy lento (0,05 ambos), solo hacia una iteración


"""

from sklearn.model_selection import train_test_split, cross_val_score

### No hará falta volverlas a dividir borrar luego ###
ds = rel_Rmag_mumax
x = ds[["Rmag"]].to_numpy()
y = ds[["mumax"]].to_numpy()
# split into training, validation, and testing sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

alphas = [0.01, 0.02, 0.03, 0.04, 0.05]
lambdas = [0.01, 0.02, 0.03, 0.04, 0.05]
resultats = []

# Busquem les millors alphes i lambdes
for a in alphas:
    array_aux = []
    for lam in lambdas:
        regr = Regressor(-10, -10, a, lam)
        regr.train(100, x_train, y_train)
        aux = regr.predict(x_test)
        array_aux.append(mean_squeared_error(aux, y_test))
        #print("alpha: "+str(a)+", lambda: "+str(lam)+", mse: " + str(mean_squeared_error(aux, y_test)))
    resultats.append(array_aux)
    print(array_aux)

# el regresor funciona mejor cuando lamba es 0.03. Vuelven a subir porque lamba es peor en ese caso.
# vemos que las mejores son con lamba 0.03 y escogemos o alpha 0.04 o 0.05 -> hacer cross validation para comprobar que
# el resultado funciona siempre y no sea coincidencia

# hacemos k-fold cross_validation para comprobar que no sea coincidencia


# Gràfica de la evolución del coste del regressor en avanzar el numero de iteracions (8)
reg34 = Regressor(-10, -10, 0.04, 0.03)
reg34.train(100, x_train, y_train)
plot_cost = plt.figure()
xax = [i for i in range(0, 8)]
plt.plot(xax, reg34.cost, color='#d2705b')
plt.show()

# Fem cross validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=1, shuffle=True)
scores = []
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = Regressor(-10, -10, 0.04, 0.03)
    model.train(100, x_train, y_train)
    scores.append(mean_squeared_error(y_test, model.predict(x_test) ))

print("Scores: ", scores)
print("Mean:", np.mean(scores))

# El cross validation me dice si funciona bien auqnue cualquier dataset de train y test. Vemos que los resultados son
# consistentes, porque se mantienen igual que en el test de las mejores alphas y lambdas
# el valor es el mse, el regressor me dice el valor de la prediccion y en el mse lo comparo con el test.
# Por cada prediccion me equivoco minerror = raiz(0.175) = 0,42 aproximadamente, se equivoca en un 2,17%.
#

