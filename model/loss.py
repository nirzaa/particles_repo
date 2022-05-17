import torch.nn.functional as F
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

def loss_3_5(output, target):
    return criterion(output, target)

def nll_loss(output, target):
    return F.nll_loss(output, target)


def MSE_loss(output, target):
    return F.mse_loss(output, target)

def my_loss(output, target):
    loss = ((output - target) / target)**2
    loss = loss.mean()
    return loss
