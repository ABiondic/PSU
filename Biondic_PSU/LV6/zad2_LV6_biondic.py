from sklearn import datasets
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def generate_data(n_samples, flagc):

    if flagc == 1:
        random_state = 365
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)

    elif flagc == 2:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                          centers=4,
                          cluster_std=[1.0, 2.5, 0.5, 3.0],
                          random_state=random_state)

    elif flagc == 4:
        X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)

    elif flagc == 5:
        X, y = datasets.make_moons(n_samples=n_samples, noise=.05)

    else:
        X = []

    return X

X = generate_data(500, 1)

inertia_values = []

for k in range(1, 21):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 21), inertia_values, marker='o')
plt.title("Zadatak 2 - Elbow metoda")
plt.xlabel("Broj klastera")
plt.ylabel("Vrijednost kriterijske funkcije")
plt.xticks(range(1, 21))
plt.grid(True)
plt.show()

for k, value in zip(range(1, 21), inertia_values):
    print(f"Broj klastera: {k}, kriterijska funkcija: {value}")


"""
U drugom zadatku vrijednost kriterijske funkcije smanjuje se kako se povećava broj klastera.
To mozemo ocekivati jer se podaci dijele u sve manje grupe, pa su udaljenosti podataka od centara manje.
Optimalan broj klastera može se odrediti pomocu takozvane elbow metode.
Na grafu se trazi mjesto gdje se pad vrijednosti kriterijske funkcije pocinje usporavati
,a to mjesto izgleda kao “lakat” i predstavlja dobar izbor broja klastera.

"""