import torch
import torch.nn as nn
import torch.nn.functional as F

def preturb(    model : nn.Module,
                x_org : torch.Tensor,
                x : torch.Tensor,
                y : torch.Tensor,
                epsilon : float,
                alpha : float, val_max : float, val_min : float) -> torch.Tensor :
    x.requires_grad = True
    model.eval()
    with torch.enable_grad() :
        y_pred = model(x, eval=True)
        loss = F.cross_entropy(y_pred, y)
        grad = torch.autograd.grad(loss, x, only_inputs=True)[0]
        x.data += alpha * torch.sign(grad.data)
        
        # apply linf
        min_x = x_org - epsilon
        max_x = x_org + epsilon
        x = torch.max(torch.min(x, max_x), min_x)
        x.clamp_(val_min, val_max)

    model.train()
    return x.data