import torch
import torch.nn as nn
# import numpy as np

class PMDLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(PMDLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input, target, reduce=True):
        batch_size = input.size()[0]

        input = input.view(batch_size, -1)
        target = target.view(batch_size, -1).float()
        
        target_positive_location = target.clone()
        target_positive_location[target_positive_location>0]=1
        input = input * target_positive_location

        # input = torch.unsqueeze(input,0)
        # target = torch.unsqueeze(target,0)

        # input_target = torch.cat([input, target], dim=0)

        # rmin = torch.min(input_target,dim=0)[0]
        # rmax = torch.max(input_target,dim=0)[0]

        pmd_loss = []
        for idx in range(batch_size):
            # L1
            iou_loss_sub = torch.abs(input[idx]-target[idx])

            # #L2
            # iou_loss_sub = iou_loss_sub*iou_loss_sub


            iou_loss_sub = iou_loss_sub[target[idx]>0]
            if torch.sum(iou_loss_sub)>0:
                iou_loss_sub = torch.mean(iou_loss_sub)
                pmd_loss.append(iou_loss_sub)
            else:
                pmd_loss.append(torch.mean(target)*0)


        if len(pmd_loss)==0:
            pmd_loss.append(torch.mean(target)*0)
            pmd_loss = self.loss_weight * torch.stack(pmd_loss, dim=0)
        else:
            pmd_loss = self.loss_weight * torch.stack(pmd_loss, dim=0)
        
        if reduce:
            pmd_loss = torch.mean(pmd_loss)

        return pmd_loss
