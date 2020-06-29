from itertools import count
import torch
import torch.autograd
import torch.nn.functional as F

PLOY_DEGREE = 3

def make_feature(x):
    x = x.unsqueeze(1)
    return torch.cat([x**i for i in range(1, PLOY_DEGREE+1)], 1)

W_target = torch.randn(PLOY_DEGREE,1)
b_target = torch.randn(1)

def f(x):
    return x.mm(W_target) + b_target.item()

def get_batch(batch_size = 32):
    random = torch.randn(batch_size)
    x = make_feature(random)
    y = f(x)
    return x,y
#defind Model
fc = torch.nn.Linear(W_target.size(0),1)

for batch_idx in count(1):
    batch_x, batch_y = get_batch()
    fc.zero_grad()

    output = F.smooth_l1_loss(fc(batch_x),batch_y)
    loss = output.item()
    output.backward()
    #apply gradiens
    for param in fc.parameters():
        param.data.add_(-0.1*param.grad.data)

    if loss<1e-3:
        break

def poly_desc(W,b):
    result = 'y='
    for i,w in enumerate(W):
        result += '{:+.2f}X^{}'.format(w,len(W)-i)
    result += '{:+.2f}'.format(b[0])
    return result
print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
print('==> Learned function:\t' + poly_desc(fc.weight.view(-1), fc.bias))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))