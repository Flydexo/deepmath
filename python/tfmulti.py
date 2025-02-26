from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

model = Sequential()

model.add(Dense(3, input_dim=2, activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))

model.layers[0].set_weights(
    [
        np.array([[1., 3., -5.], [2., -4., -6.]]),
        np.array([-1., 0., 1.])
    ]
)
model.layers[1].set_weights([np.array([[1.], [1.], [1.]]), np.array([-3])])

input = np.array([[7, -5]])
output = model.predict(input)
print(output)

from mpl_toolkits.mplot3d import Axes3D

VX = np.linspace(-5, 5, 30)
VY = np.linspace(-5, 5, 30)
X,Y = np.meshgrid(VX, VY)

input = np.c_[X.ravel(),Y.ravel()]
output = model.predict(input)
Z = output.reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title('3D Plot of Model Predictions')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()

