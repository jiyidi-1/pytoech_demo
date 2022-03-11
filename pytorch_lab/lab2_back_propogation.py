import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1, 2, 3]
y_data = [2, 4, 6]

w1 = torch.tensor([2.0], requires_grad=True)
w2 = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([4.0], requires_grad=True)


def forward(x_f):
    return w1 * x_f ** 2 + w2 * x_f + b


def loss(x_l, y_l):
    y_pred = forward(x_l)
    return (y_pred - y_l) ** 2


print("Predict(before training)", 4, forward(4).item())
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        # print('\tw1 grad:', w1.grad.item())
        # print('\tw2 grad:', w2.grad.item())
        # print('\tb grad:', b.grad.item())
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data

        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print('progress:', epoch, l.item())

# print("w1 =", w1.data, "w2 =", w2.data, "b =", b.data)
print("Predict(after training)", 4, forward(4).item())

x_plot = np.arange(-20, 21)
y_plot = []
for x in x_plot:
    y_plot.append(forward(x).detach().numpy())
plt.plot(x_plot, y_plot)
plt.xlim(-20, 20)
plt.show()
