import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

# 损失函数
criterion = torch.nn.MSELoss()

# 优化器
optimizer1 = torch.optim.SGD(model.parameters(), lr=0.01)
# optimizer2 = torch.optim.Adagrad(model.parameters(), lr=0.01)
# optimizer3 = torch.optim.Adam(model.parameters(), lr=0.01)
# optimizer4 = torch.optim.Adamax(model.parameters(), lr=0.01)
# optimizer5 = torch.optim.ASGD(model.parameters(), lr=0.01)
# optimizer6 = torch.optim.LBFGS(model.parameters(), lr=0.01)
# optimizer7 = torch.optim.RMSprop(model.parameters(), lr=0.01)
# optimizer8 = torch.optim.Rprop(model.parameters(), lr=0.01)


for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    # 测试每一种优化器

    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()
    #
    # optimizer2.zero_grad()
    # loss.backward()
    # optimizer2.step()
    #
    # optimizer3.zero_grad()
    # loss.backward()
    # optimizer3.step()
    #
    # optimizer4.zero_grad()
    # loss.backward()
    # optimizer4.step()
    #
    # optimizer5.zero_grad()
    # loss.backward()
    # optimizer5.step()
    #
    # optimizer6.zero_grad()
    # loss.backward()
    # optimizer6.step()
    #
    # optimizer7.zero_grad()
    # loss.backward()
    # optimizer7.step()
    #
    # optimizer8.zero_grad()
    # loss.backward()
    # optimizer8.step()


print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
