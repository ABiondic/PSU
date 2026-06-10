from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_gray = mpimg.imread("example_grayscale.png")

if len(image_gray.shape) == 3:
    image_gray = image_gray[:, :, 0]

X_img = image_gray.reshape((-1, 1))

k = 10

kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
kmeans.fit(X_img)

values = kmeans.cluster_centers_.squeeze()
labels = kmeans.labels_

image_compressed = np.choose(labels, values)
image_compressed.shape = image_gray.shape

plt.figure()
plt.imshow(image_gray, cmap='gray')
plt.title("Originalna grayscale slika")
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(image_compressed, cmap='gray')
plt.title(f"Kvantizirana grayscale slika, K = {k}")
plt.axis("off")
plt.show()



height, width = image_gray.shape

original_bits = height * width * 8

bits_per_label = int(np.ceil(np.log2(k)))

compressed_bits = height * width * bits_per_label + k * 8

compression_ratio = original_bits / compressed_bits

print("Dimenzije slike:", image_gray.shape)
print("Broj klastera:", k)
print("Broj bitova originalne slike:", original_bits)
print("Broj bitova komprimirane slike:", compressed_bits)
print("Omjer kompresije:", compression_ratio)



"""

Kod kvantizirane grayscale slike s K = 10 vidimo da slika ima manje nijansi sive nego originalna slika.
Prijelazi između svijetlih i tamnih područja su grublji, a na slici se pojavljuju veće površine iste nijanse.
Unatoč tome, glavni sadržaj slike ostaje prepoznatljiv.
Povećanjem broja klastera kvantizirana slika bila bi sličnija originalu, dok bi smanjenjem broja klastera gubitak detalja bio veći.

"""