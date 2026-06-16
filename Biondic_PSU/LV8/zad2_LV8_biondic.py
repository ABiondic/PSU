import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from skimage import color
from tensorflow.keras import models
import numpy as np

filename = 'test.png'

img_original = mpimg.imread(filename)


if img_original.ndim == 3 and img_original.shape[2] == 4:
    img_original = img_original[:, :, :3]


if img_original.ndim == 3:
    img = color.rgb2gray(img_original)
else:
    img = img_original

img = resize(img, (28, 28))

if np.mean(img) > 0.5:
    img = 1 - img


plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.show()


img = img.reshape(1, 28, 28, 1)
img = img.astype('float32')

model = models.load_model('best_model.h5')


prediction = model.predict(img)


predicted_class = np.argmax(prediction)

print("Vjerojatnosti po klasama:")
for i, prob in enumerate(prediction[0]):
    print(f"Znamenka {i}: {prob:.4f}")

print(f"\nModel predvida da je na slici znamenka: {predicted_class}")