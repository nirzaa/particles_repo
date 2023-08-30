import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)


def MSE_loss(output, target):
    return F.mse_loss(output, target)

def my_loss(output, target):
    loss = ((output - target) / target)**2
    loss = loss.mean()
    return loss

def normalized_loss(output, target):
    output = torch.clamp(output, min=1e-8, max=float('inf'))
    target = torch.clamp(target, min=1e-8, max=float('inf'))
    loss = (1 - output/target)**2
    loss = loss.mean()
    return loss

def david_loss(output, target):
    # loss = (1 - output/target)**2
    loss = (1 - output/(target+0.000000001))**2
    loss = loss.mean()
    return loss
