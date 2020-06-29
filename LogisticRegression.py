import torch
import matplotlib.pyplot as plt
from torch import nn
from itertools import count
from torch.autograd import Variable
import numpy as np
'''
逻辑斯蒂回归是线性模型的概率表达方式，其核心含义在于二分类问题的几率log(p/(1-p))为线性模型，其中概率p的表达为p = 1/(1+ exp(wx+b)),
因此可以将逻辑logistics回归分解为两部分线性分类器+概率表达式
'''


#Make Data
n_data = torch.ones(100, 2 )
x0 = torch.normal(2*n_data,1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data,1) #data
y1 = torch.ones(100) #label

x = torch.cat((x0,x1), 0).type(torch.FloatTensor)
y = torch.cat((y0,y1), 0).type(torch.FloatTensor)
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:,1], c=y.data.numpy(),s=100,lw=0,cmap='RdYlGn')
# plt.show()

##Defind the model ,loss function and optimizer
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.lr = nn.Linear(2,1)#Liner function: two inputs 1 outputs
        self.sm = nn.Sigmoid()
    def forward(self,x):
        x = self.lr(x)
        x = self.sm(x)
        return x

#Check First
if torch.cuda.is_available():
    Model = LogisticRegression.cuda()
else:
    Model = LogisticRegression()

LossFunction = nn.BCELoss()
Optimizer = torch.optim.SGD(Model.parameters(), lr=1e-3, momentum=0.9) #momentum 动量参数

#Train

for epoch in count(1):
    if torch.cuda.is_available():
        x_data = Variable(x).cuda()
        y_data = Variable(y).cuda()
    else:
        x_data = Variable(x)
        y_data = Variable(y)

    out = Model.forward(x_data)
    loss = LossFunction(out, y_data)
    print_loss = loss.data.item()
    mask = out.ge(0.5).float() #以0.5对输出结果进行分类
    correct = (mask[:, 0] == y_data).sum()#正确样本数
    acc = correct.data.item()/x_data.size(0)
    Optimizer.zero_grad()
    loss.backward()
    Optimizer.step()
    if (epoch + 1) % 20 == 0:
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))  # 训练轮数
        print('loss is {:.4f}'.format(print_loss))  # 误差
        print('acc is {:.4f}'.format(acc))  # 精度
    if print_loss < 1e-2:
        break

# 结果可视化
w0, w1 = Model.lr.weight[0]
w0 = float(w0.item())
w1 = float(w1.item())
b = float(Model.lr.bias.item())
plot_x = np.arange(-7, 7, 0.1)
plot_y = (-w0 * plot_x - b) / w1
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.plot(plot_x, plot_y)
plt.show()
























