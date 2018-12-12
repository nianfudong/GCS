import torch
import torch.nn as nn
import torch.nn.functional as func
import math

class Wing_loss(nn.Module):
    def __init__(self):
        super(Wing_loss,self).__init__()

    def forward(self,pred,truth,w=10.0,epsilon=2.0):
        x=truth-pred
        c=w*(1.0-math.log(1.0+w/epsilon))
        absolute_x=torch.abs(x)
        losses=torch.where(w>absolute_x,w*torch.log(1.0+absolute_x/epsilon),absolute_x-c)

        return torch.sum(losses)/(len(losses)*1.0)