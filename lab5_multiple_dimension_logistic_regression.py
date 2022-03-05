import numpy as np
import torch
import matplotlib.pyplot as plt

'''
本期的例子使用的是糖尿病患者的情况预测，以患者的八个参数为输入，进行二分类预测其有没有的糖尿病
'''

# 加载糖尿病数据集
xy = np.loadtxt("diabetes.csv", delimiter=',', dtype=np.float32)  # 由于gpu一般使用float32，所以这里的数据类型选择float32
# 复杂的切片操作需要自己去学
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, -1:])


# 把函数分成三层进行训练
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

# 损失函数和优化器
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 玩一把大的，直接训练一百万次
for epoch in range(1000000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    # print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 在训练过10000次之后进行一次输出，查看当前的loss和准确率acc
    # 准确率acc的计算先将y_pred按照是否大于0.5归置为1和0，再将归置出来的y_pred_label与y_data进行比较，算出相同的个数做成准确率
    if epoch % 10000 == 9999:
        y_pred_label = torch.where(y_pred >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))

        acc = torch.eq(y_pred_label, y_data).sum().item() / y_data.size(0)
        print(epoch, "  loss = ", loss.item(), "acc = ", acc)
