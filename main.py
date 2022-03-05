import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 10000)
y = []


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


for x_item in x:
    y.append(sigmoid(x_item))

plt.plot(x, y)
plt.show()
