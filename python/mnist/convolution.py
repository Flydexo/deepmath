import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(X_train_data, Y_train_data), (X_test_data, Y_test_data) = mnist.load_data()
N = X_train_data.shape[0]
X_train = np.reshape(X_train_data, (N,28,28,1))
X_test = np.reshape(X_test_data, (10000,28,28,1))
X_train = X_train/255 
X_test = X_test/255
Y_train = to_categorical(Y_train_data, num_classes=10)
Y_test = to_categorical(Y_test_data, num_classes=10)

model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding='same', activation="relu", input_shape=(28,28,1)))
model.add(Conv2D(16, kernel_size=3, padding='same', activation="relu", input_shape=(28,28,32)))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',  
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, epochs=5, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])