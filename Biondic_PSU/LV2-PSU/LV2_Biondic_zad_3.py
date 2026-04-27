import numpy as np
import matplotlib.pyplot as plt


img = plt.imread("tiger.png")

img = img[:, :, 0].copy()

plt.figure()
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.show()


# POSVIJETLITI SLIKU (brightness)
bright = img + 50  
bright = np.clip(bright, 0, 255)  

plt.figure()
plt.title("Posvijetljena slika")
plt.imshow(bright, cmap="gray")
plt.show()


# ROTACIJA 
rotated = np.rot90(img, k=-1) 

plt.figure()
plt.title("Rotirana slika")
plt.imshow(rotated, cmap="gray")
plt.show()


# c) ZRCALJENJE (horizontalno)
mirrored = np.fliplr(img)

plt.figure()
plt.title("Zrcaljena slika")
plt.imshow(mirrored, cmap="gray")
plt.show()


# d) SMANJENJE REZOLUCIJE 
factor = 10
reduced = img[::factor, ::factor]

plt.figure()
plt.title("Smanjena rezolucija")
plt.imshow(reduced, cmap="gray")
plt.show()


# SAMO DRUGA ČETVRTINA PO ŠIRINI
h, w = img.shape

result = np.zeros_like(img)

result[:, w//4:w//2] = img[:, w//4:w//2]

plt.figure()
plt.title("Druga četvrtina slike")
plt.imshow(result, cmap="gray")
plt.show()