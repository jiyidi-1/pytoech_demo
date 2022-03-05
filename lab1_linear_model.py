import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# 前馈计算
def forward(x):
    return w * x + b


# 损失函数
def loss(x, y):
    return (forward(x) - y) ** 2


w_list = np.arange(0, 4.1, 0.1)
b_list = np.arange(-2.0, 2.1, 0.1)

# 这里通过函数meshgrid()把两个list组成一个网格，代表41*41个点，其中点的横纵坐标分别放在w与b两个二维数组中的对应位置
[w, b] = np.meshgrid(w_list, b_list)
# print(w, b)

loss_sum = 0
for x_val, y_val in zip(x_data, y_data):
    y_val_pred = forward(x_val)
    loss_val = loss(x_val, y_val)
    loss_sum += loss_val
    # print('\t', x_val, y_val, y_val_pred, loss_val)
# print("MSE =", loss_sum / 3)


fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(w, b, loss_sum / 3)
plt.show()
