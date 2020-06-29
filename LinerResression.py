import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = 3*x + 10+torch.rand(x.size())

class LinerRegression(nn.Module):
    def __init__(self):
        super(LinerRegression,self).__init__()
        self.liner = nn.Linear(1,1)
    def forword(self,x):
        out = self.liner(x)
        return out
if torch.cuda.is_available():
    model = LinerRegression().cuda()
else:
    model = LinerRegression()

#defind loss function and optimization function
criterion = nn.MSELoss()#均方误差为损失函数
optimizer = torch.optim.SGD(model.parameters(),lr=1e-2)#梯度下降为优化模型

num_epochs = 1000
for epoch in range(num_epochs):
    if torch.cuda.is_available():
        inputs = Variable(x).cuda()
        targets = Variable(y).cuda()
    else:
        inputs = Variable(x)
        targets = Variable(y)
    #forword brocdcast
    out = model.forword(inputs)
    loss = criterion(out, targets)
    #backword Brocdcast
    optimizer.zero_grad()#梯度清零？
    loss.backward()
    optimizer.step()

    if (epoch+1)%20 == 0:
        print('Epoch[{}/{}],loss:{:.6f}'.format(epoch+1,num_epochs,loss.item()))

    model.eval()
    if torch.cuda.is_available():
        predict = model.forword(Variable(x).cuda())
        predict = predict.data.cpu().numpy()
    else:
        predict = model.forword(Variable(x))
        predict = predict.data.numpy()

plt.plot(x.numpy(), y.numpy(), 'ro', label='Original Data')
plt.plot(x.numpy(), predict, label='Fitting Line')
plt.show()


