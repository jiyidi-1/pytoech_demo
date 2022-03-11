# 对于cuda的加速效果展示

import torch
import time

# 建立两个随机矩阵
a = torch.randn(10000, 1000)
b = torch.randn(1000, 10000)

# cpu 模式下的矩阵乘法计算时间
t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print("device = ", a.device, "; time = ", t1 - t0, c.norm(2))

# a,b移交到cuda（GPU）上
device = torch.device('cuda')
a = a.to(device)
b = b.to(device)
# 第一遍有cuda初始化的时间
t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print("device =", a.device, "; time =", t1 - t0, c.norm(2))

t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print("device =", a.device, "; time =", t1 - t0, c.norm(2))
