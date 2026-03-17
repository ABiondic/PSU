import numpy as np
import matplotlib.pyplot as plt


img = plt.imread("tiger.png")
img = img.rotate(90)
print(img.shape)
print(img.dtype)
plt.figure()
plt.imshow(img, cmap="gray")
plt.show()
