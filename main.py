import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 10000)
y = []


def sigmoid(x_in):
    return 1 / (1 + np.exp(-x_in))


for x_item in x:
    y.append(sigmoid(x_item))

plt.plot(x, y)
plt.show()
