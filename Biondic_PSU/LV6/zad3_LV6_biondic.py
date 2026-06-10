from sklearn import datasets
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

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

metode = ['single', 'complete', 'average', 'ward']

for metoda in metode:

    Z = linkage(X, method=metoda)

    plt.figure(figsize=(12, 6))
    dendrogram(Z)
    plt.title(f"Zadatak 3 - Dendrogram, metoda {metoda}")
    plt.xlabel("Podaci")
    plt.ylabel("Udaljenost")
    plt.show()

    """
    Dendrogram nam prikazuje spajanje sličnih podataka u veće grupe.
    Metoda ward daje nam jasnije i kompaktnije klastere jer minimizira varijancu unutar grupa.
    Metoda single s druge strane nam daje lošije rezultate jer spaja klastere preko pojedinačnih bliskih točaka.
    Metode complete i average daju rezultate između tih pristupa. Promjenom metode mijenja se način računanja udaljenosti između klastera,
    pa se mijenja i izgled dendrograma.
    
    """