import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import shapiro
import seaborn as sns
import math
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import time
__author__ = "1528873, 1525184, 1527280"


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

# Ajustem el dataset a recomonacions vistes a la documentació de Kaggle
dataset = dataset.rename(columns={"e.Mcz": "eMcz"})
dataset = dataset.drop(dataset[dataset.chi2red > 5].index)
dataset = dataset.drop(dataset[dataset.eMcz > 0.2].index)
dataset = dataset.drop(dataset[dataset.ApDRmag < 0].index)

# Creem una llista per seleccionar els nombres de columnes que volem eliminar
dataTypeDict = dict(dataset.dtypes)
lst = []
for x in dataTypeDict:
    lst.append(x)

# Drop de la columna 1 que conté l'índex
dataset = dataset.drop(columns='Nr')

# Treiem les columnes entre la 56 y la 65 ja que són valors redundants de les 13 anteriors
lst2 = lst[-10:]

# Eliminem valors inexistents
dataset = dataset.dropna()

# Drop de las columnes que donen informació sobre els errors.
for x in lst:
    if "e" in x:
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


# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
correlacio = dataset.corr()

plt.figure(figsize=(16, 5))
ax = sns.heatmap(correlacio, annot=True, linewidths=.5)
plt.show()

# Comprovem la correlació entre aquestes varibales
relacio = sns.pairplot(dataset[["Rmag", "ApDRmag", "Mcz", "mumax"]])

# Comprovem la correlació entre aquestes dos dades mitjançant gràfiques, ja que a la documentació és menciona
# la relació entre aquestes per calcular el tamany d'una estrella
rel_Rmag_mumax = dataset[["Rmag","mumax"]]
relacio2 = sns.pairplot(rel_Rmag_mumax)
plt.show()

# Si bé la corrleació és bona, es necessari normalitrzar les dades per ferles més consistents.
# Eliminem els outliers amb z_score, que ens indica quant lluny es troba cada dada de la mitjana dels atributs
from scipy import stats
rel_Rmag_mumax2 = rel_Rmag_mumax[(np.abs(stats.zscore(rel_Rmag_mumax)) < 2).all(axis=1)]

# Comprovem la correlació entre aquestes dades per mitjà de gràfiques
relacio3 = sns.pairplot(rel_Rmag_mumax2)
plt.show()

# Veiem que segueixen sense seguir distribucions Gaussianes, pero hem tret outliers


### APARTAT (B): PRIMERES REGRESSIONS

# CÁLCULO DEL ERROR CUADRÁTICO

def mean_squared_error(y1, y2):
    # comprovem que y1 i y2 tenen la mateixa mida
    assert(len(y1) == len(y2))
    mse = np.sum((y1-y2)**2)
    return mse / len(y1)

def regression(x, y):
    # Creem un objecte de regressió de sklearn
    regr = LinearRegression()
    # Entrenem el model per a predir y a partir de x
    regr.fit(x, y)
    # Retornem el model entrenat
    return regr

def predict_with_train(x, y, test_size):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    model = regression(x_train, y_train)
    y_hat = model.predict(x_test)

    # Mostramos el resultado
    plt.figure()
    plt.plot(x_train, y_train, '-o', alpha = 0.25)
    plt.plot(x_test, y_hat, 'r', alpha = 0.25)
    plt.xlabel('mumax')
    plt.ylabel('')

    print ("MSE:", mean_squared_error(y_hat, y_test))

# Per veure quins atributs tenen un menor MSE
x = dataset[["Rmag"]]
y = dataset[["mumax"]]
print(predict_with_train(x, y, test_size=0.33))

"""
### APARTAT (A): EL DESCENS DEL GRADIENT
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
        # print((self.w1 * x) + self.w0)
        return (self.w1 * x) + self.w0

    def __update(self, dy, x):
        # actualitzar aqui els pesos donada la prediccio (dy) i la y real.
        self.w0 = self.w0 - ((self.alpha/len(dy)) * np.sum(dy))
        m = (self.alpha * self.lamda / len(dy))
        self.w1 = self.w1 * (1 - m) - m * np.sum(dy * x)


    def train(self, max_iter, x, y):
        # Entrenar durant max_iter iteracions o fins que la millora sigui inferior a epsilon
        jw_prev = 10000000
        for i in range(0, max_iter):
            hy = self.predict(x)
            diff_hy = hy - y
            jw = (1 / (2 * len(diff_hy))) * np.sum(diff_hy ** 2) + (self.lamda * ((self.w0 ** 2) + (self.w1 ** 2)))
            self.cost.append(jw)
            if (np.abs(jw - jw_prev) / jw_prev) < 0.05:
                break
            self.n_iteracions += 1
            self.__update(diff_hy, x)
            jw_prev = jw


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
        array_aux.append(mean_squared_error(aux, y_test))
        # print("alpha: "+str(a)+", lambda: "+str(lam)+", mse: " + str(mean_squared_error(aux, y_test)))
    resultats.append(array_aux)
    print(array_aux)

# Veiem que el regressor funciona millor amb lambda 0,03 i alpha 0,04/0,05 (veure taula)

# Apliquem Cross Validation amb diferents sub-conjunts.
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=1, shuffle=True)
scores = []
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = Regressor(-10, -10, 0.03, 0.03)
    model.train(100, x_train, y_train)
    scores.append(mean_squared_error(y_test, model.predict(x_test)))

print("Scores: ", scores)
print("Mean:", np.mean(scores))

# Gràfica de la evolució del cost del regressor en avançar el nombre de iteracions (8)
start = time.time()
reg34 = Regressor(-10, -10, 0.03, 0.03)
reg34.train(100, x_train, y_train)
end = time.time()
dif = end - start
print("Temps nostre regressor: ", dif)
plot_cost = plt.figure()
xax = [i for i in range(0, 10)]
plt.plot(xax, reg34.cost, color='#d2705b')
plt.title('Evolució del cost')
plt.xlabel("Nombre d'iteracions")
plt.ylabel("Cost")
plt.show()

# Comparació entre el nostre regressor i el de la llibreria
start = time.time()
reg = LinearRegression()
reg.fit(x_train, y_train)
pred = reg.predict(x_test)
end = time.time()
dif = end - start
print("Score llibreria: ", mean_squared_error(pred, y_test), "Temps regressor llibreria: ", dif)
"""
