from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

def heaviside(x):
    if x < 0:
        return 0
    else:
        return 1

model = Sequential()

model.add(Dense(2, input_dim=1, activation=heaviside))
model.add(Dense(1, activation=heaviside))

model.layers[0].set_weights([np.array([[1., -0.5]]), np.array([-1,1])])

model.layers[1].set_weights([np.array([[1.0], [1.0]]), np.array([0])])

input = np.array([[3.0]])
output = model.predict(input)
print('input: ',input, ' output: ', output)

import matplotlib.pyplot as plt

X = np.linspace(-2, 3, num=100)
input = np.array([[x] for x in X])
output = model.predict(input)
Y = np.array([y[0] for y in output])
plt.plot(X,Y)
plt.tight_layout()
plt.show()