import torch
from scipy.stats import norm


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        print(output.shape)
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def bias_avg(output, target):
    bias = target - output
    with torch.no_grad():
        mean, std = norm.fit(bias.cpu())
    return mean


def bias_std(output, target):
    bias = target - output
    with torch.no_grad():
        mean, std = norm.fit(bias.cpu())
    return std


def bias(output, target):
    with torch.no_grad():
        bias = target - output
    return bias