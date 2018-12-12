import torch
import torch.nn as nn
import torch.nn.functional as func
import math

class Gcs_loss(nn.Module):
    def __init__(self):
        super(Gcs_loss,self).__init__()

    def forward(self,pred,truth):
        batchSize = int(pred.size()[0])
        pointsNum = int(pred.size()[1]/2)
        #print(pointsNum)
        predX = pred[:,0:pointsNum]
        predY = pred[:,pointsNum:]

        truthX = truth[:, 0:pointsNum]
        truthY = truth[:, pointsNum:]

        losses = []
        for ix in range(0,pointsNum):
            expandPredX = predX[:, ix].view(-1,1).expand(batchSize, pointsNum)
            diffPredX = predX - expandPredX
            expandTruthX = truthX[:, ix].view(-1, 1).expand(batchSize, pointsNum)
            diffTruthX = truthX - expandTruthX
            distX = torch.dist(diffPredX, diffTruthX, 2)

            expandPredY = predY[:, ix].view(-1, 1).expand(batchSize, pointsNum)
            diffPredY = predY - expandPredY
            expandTruthY = truthY[:, ix].view(-1, 1).expand(batchSize, pointsNum)
            diffTruthY = truthY - expandTruthY
            distY = torch.dist(diffPredY, diffTruthY, 2)

            losses.append(distX)
            losses.append(distY)
        losses = torch.Tensor(losses).cuda()
        #print(pred)
        #print(predY)
        #print(predY.size())

        return torch.sum(losses)/(batchSize*2.0)