import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 10000)
y = []


def sigmoid(x_in):
    return 1 / (1 + np.exp(-x_in))


if __name__ == "__main__":
    for x_item in x:
        y.append(sigmoid(x_item))

    zero = np.zeros(10000)
    one = np.ones(10000)
    plt.plot(x, y)
    plt.plot(x, zero)
    plt.plot(x, one)
    plt.show()
