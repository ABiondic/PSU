# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


def non_func(x):
    y = (
        1.6345
        - 0.6235 * np.cos(0.6067 * x)
        - 1.3501 * np.sin(0.6067 * x)
        - 1.1622 * np.cos(2 * x * 0.6067)
        - 0.9443 * np.sin(2 * x * 0.6067)
    )
    return y


def add_noise(y):
    np.random.seed(14)
    varNoise = np.max(y) - np.min(y)
    y_noisy = y + 0.1 * varNoise * np.random.normal(0, 1, len(y))
    return y_noisy


def run_model(n_samples):
    x = np.linspace(1, 10, n_samples)
    y_true = non_func(x)
    y_measured = add_noise(y_true)

    x = x[:, np.newaxis]
    y_measured = y_measured[:, np.newaxis]

    np.random.seed(12)
    indeksi = np.random.permutation(len(x))

    indeksi_train = indeksi[0:int(np.floor(0.7 * len(x)))]
    indeksi_test = indeksi[int(np.floor(0.7 * len(x))) + 1:len(x)]

    degrees = [2, 6, 15]
    MSEtrain = []
    MSEtest = []

    plt.figure()
    plt.plot(x, y_true, label="f")
    plt.plot(x[indeksi_train], y_measured[indeksi_train], "ok", label="train")

    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)
        xnew = poly.fit_transform(x)

        xtrain = xnew[indeksi_train, :]
        ytrain = y_measured[indeksi_train]

        xtest = xnew[indeksi_test, :]
        ytest = y_measured[indeksi_test]

        linearModel = lm.LinearRegression()
        linearModel.fit(xtrain, ytrain)

        ytrain_p = linearModel.predict(xtrain)
        ytest_p = linearModel.predict(xtest)

        MSEtrain.append(mean_squared_error(ytrain, ytrain_p))
        MSEtest.append(mean_squared_error(ytest, ytest_p))

        plt.plot(x, linearModel.predict(xnew), label="degree=" + str(degree))

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Usporedba modela za n = " + str(n_samples))
    plt.legend(loc=4)

    return np.array(MSEtrain), np.array(MSEtest)


# Glavni dio zadatka
MSEtrain, MSEtest = run_model(50)

print("Za degree = 2, 6, 15:")
print("MSEtrain =", MSEtrain)
print("MSEtest =", MSEtest)


# Simulacija za manji i veći broj uzoraka
print("\nSimulacija za razlicit broj uzoraka:")

for n in [20, 50, 100, 200]:
    mse_tr, mse_te = run_model(n)
    print("\nBroj uzoraka:", n)
    print("MSEtrain =", mse_tr)
    print("MSEtest =", mse_te)


#Komentar:
#Kod manjeg broja uzoraka model ima manje podataka za učenje.
#Zbog toga je prenaučenost izraženija, posebno kod degree=15.
#Kod većeg broja uzoraka rezultati su stabilniji.
#Model degree=2 je jednostavan i može biti podnaučen.
#Model degree=6 obično bolje prati funkciju.
#Model degree=15 može imati malu pogrešku na train skupu, ali veću na test skupu.

plt.show()