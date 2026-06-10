from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



image_color = mpimg.imread("example.png")

if image_color.shape[2] == 4:
    image_color = image_color[:, :, :3]

height, width, channels = image_color.shape

X_img = image_color.reshape((-1, 3))

k = 10

kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
kmeans.fit(X_img)

values = kmeans.cluster_centers_
labels = kmeans.labels_

image_compressed = values[labels]
image_compressed = image_compressed.reshape((height, width, channels))

plt.figure()
plt.imshow(image_color)
plt.title("Originalna slika")
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(image_compressed)
plt.title(f"Kvantizirana slika u boji, K = {k}")
plt.axis("off")
plt.show()