import torch.nn.functional as F
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

def loss_3_5(output, target):
    try:
        return criterion(output, target)
    except:
        return criterion(output, target.argmax(axis=1))

def loss_3_5_reg(output, target):
    return F.mse_loss(output, target.argmax(axis=1).float().unsqueeze(axis=1))

def nll_loss(output, target):
    return F.nll_loss(output, target)


def MSE_loss(output, target):
    return F.mse_loss(output, target)

def my_loss(output, target):
    loss = ((output - target) / target)**2
    loss = loss.mean()
    return loss
