import torch
import torch.nn.functional as F

def MSE_LOSS(rgb, rgb0, target):
    loss = F.mse_loss(rgb, target)
    loss0 = F.mse_loss(rgb0, target)
    return loss + loss0

def Adaptive_MSE_LOSS(rgb, rgb0, target):
    weights = torch.sqrt(torch.abs(rgb - target)).detach()
    loss = F.mse_loss(rgb, target)
    loss0 = torch.mean(weights * torch.square(rgb0 - target))
    return loss + loss0

def MSE_LOSS_REGULARISATION(rgb, params, target):
    loss = F.mse_loss(rgb, target)
    reg = torch.mean(torch.square(params))
    return loss + 0.1**6 * reg