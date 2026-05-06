import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# TODO: prikazi nekoliko slika iz train skupa

plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.axis('off')
plt.show()

# Skaliranje vrijednosti piksela na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# Slike 28x28 piksela se predstavljaju vektorom od 784 elementa
x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)

# Kodiraj labele (0, 1, ... 9) one hot encoding-om
y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)


# TODO: kreiraj mrezu pomocu keras.Sequential(); prikazi njenu strukturu pomocu .summary()

model = keras.Sequential()
model.add(layers.Dense(units=128, activation='relu', input_shape=(784,)))
model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=10, activation='softmax'))
model.summary()


# TODO: definiraj karakteristike procesa ucenja pomocu .compile()

model.compile(
    optimizer='adam',           
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# TODO: provedi treniranje mreze pomocu .fit()

model.fit(x_train_s, y_train_s, epochs=10, batch_size=32, validation_split=0.1)

# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje

train_loss, train_acc = model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_acc = model.evaluate(x_test_s, y_test_s, verbose=0)

print(f"Tocnost na skupu za ucenje: {train_acc:.4f}")
print(f"Tocnost na testnom skupu: {test_acc:.4f}")

# TODO: Prikazite matricu zabune na skupu podataka za testiranje

y_test_pred = np.argmax(model.predict(x_test_s), axis=1)
cm_test = confusion_matrix(y_test, y_test_pred)

print("Matrica zabune za testni skup:")
ConfusionMatrixDisplay(cm_test).plot(cmap='Blues')
plt.show()

# TODO: Prikazi nekoliko primjera iz testnog skupa podataka koje je izgrađena mreza pogresno klasificirala

wrong_idx = np.where(y_test_pred != y_test)[0]

plt.figure(figsize=(12, 5))
for i, idx in enumerate(wrong_idx[:10]):  
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Stvarno: {y_test[idx]}\nPred: {y_test_pred[idx]}")
    plt.axis('off')
plt.show()
