import torch
import torch.nn as nn
import torch.nn.functional as F


def CrossEntropyLoss(output, target, weights): 
    weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=weights)
    return criterion(output, target)

def CrossEntropyLossWithoutWeight(output, target): 
    criterion = nn.CrossEntropyLoss()
    return criterion(output, target)

def BinaryCrossEntropyLoss(output, target):
    return F.binary_cross_entropy(output, target)


