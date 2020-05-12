import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def load_dataset():
    arr = []
    with open("teachers.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            arr.append(row)
    new_set = []
    for row in arr:
        for j in range(0,len(row)):
            # Skip first evaluation
            if float(row[j]) == 0.0:
                continue
            if (j + 1) == len(row):
                new_set.append([round(float(row[j])), round(float(row[j]))])
            else:
                new_set.append([round(float(row[j])), round(float(row[j + 1]))])
    return new_set

def plot_dataset(dataset):
    for row in dataset:
        plt.plot(row[0], row[1], "bo")

dataset = load_dataset()

dataset_train = dataset[2435:]
dataset_test = dataset[:609]

X_train = np.array([x[0] for x in dataset_train]).reshape(-1, 1)
y_train = np.array([y[1] for y in dataset_train])

X_test = np.array([x[0] for x in dataset_test]).reshape(-1, 1)
y_test = np.array([y[1] for y in dataset_test])

regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test)
plot_dataset(dataset_test)
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

prediction = regressor.predict([[80]])
print(f"Para un maestro que tenia un promedio de 80 el ciclo pasada, predecimos {prediction[0]} basandonos\
    en las calificaciones de todos los demas maestros en la plataforma")
print(0.82619047 * 80 + 13.36874673197947)

prediction = regressor.predict([[47]])
print(f"Para un maestro que tenia un promedio de 47 el ciclo pasada, predecimos {prediction[0]} basandonos\
    en las calificaciones de todos los demas maestros en la plataforma")

prediction = regressor.predict([[90]])
print(f"Para un maestro que tenia un promedio de 90 el ciclo pasada, predecimos {prediction[0]} basandonos\
    en las calificaciones de todos los demas maestros en la plataforma")
