from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


x_train_s = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_s = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train_s = to_categorical(y_train, num_classes=10)
y_test_s = to_categorical(y_test, num_classes=10)


model = models.Sequential()

model.add(layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=(28, 28, 1)
))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation='relu'
))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(
    units=64,
    activation='relu'
))

model.add(layers.Dense(
    units=10,
    activation='softmax'
))

model.summary()


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


my_callbacks = [
    callbacks.TensorBoard(
        log_dir='logs',
        update_freq=100
    ),
    callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]


history = model.fit(
    x_train_s,
    y_train_s,
    epochs=3,
    batch_size=64,
    validation_split=0.1,
    callbacks=my_callbacks
)


best_model = keras.models.load_model('best_model.h5')


y_train_pred_prob = best_model.predict(x_train_s)
y_train_pred = np.argmax(y_train_pred_prob, axis=1)


y_test_pred_prob = best_model.predict(x_test_s)
y_test_pred = np.argmax(y_test_pred_prob, axis=1)


train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\nTocnost najboljeg modela:")
print(f"Tocnost na skupu za ucenje: {train_accuracy:.4f}")
print(f"Tocnost na skupu za testiranje: {test_accuracy:.4f}")

cm_train = confusion_matrix(y_train, y_train_pred)
print("\nMatrica zabune na skupu za ucenje:")
print(cm_train)

cm_test = confusion_matrix(y_test, y_test_pred)
print("\nMatrica zabune na skupu za testiranje:")
print(cm_test)


"""

Ako je tocnost na skupu za ucenje i testiranje visoka i slicna, model dobro generalizira.
Ako je tocnost na skupu za ucenje znatno veca nego na testnom skupu, moguce je prenaucenost modela.
Matrica zabune pokazuje koje znamenke model najcesce pogresno klasificira.

"""