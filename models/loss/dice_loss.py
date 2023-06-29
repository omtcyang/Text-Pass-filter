import torch
import torch.nn as nn


# class DiceLoss(nn.Module):
#
#     def __init__(self,
#                  use_sigmoid=True,
#                  loss_weight=1.0):
#         super(DiceLoss, self).__init__()
#         self.use_sigmoid = use_sigmoid
#         self.loss_weight = loss_weight
#
#     def forward(self, pred, target, mask, reduction='mean'):
#         if self.use_sigmoid:
#             pred = torch.sigmoid(pred)
#
#         batch_size = pred.size(0)
#
#         pred = pred.contiguous().view(batch_size, -1)
#         target = target.contiguous().view(batch_size, -1).float()
#         mask = mask.contiguous().view(batch_size, -1).float()
#
#         pred = pred * mask
#         target = target * mask
#
#         a = torch.sum(pred * target, dim=1)
#         b = torch.sum(pred * pred, dim=1) + 0.001
#         c = torch.sum(target * target, dim=1) + 0.001
#         d = (2 * a) / (b + c)
#         dice_loss = 1 - d
#
#         dice_loss = self.loss_weight * dice_loss
#
#         if reduction == 'mean':
#             dice_loss = torch.mean(dice_loss)
#
#         return dice_loss

class DiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self, pred, target, ignore, reduction='mean'):
        if self.use_sigmoid:
            pred = torch.sigmoid(pred)
        if ignore is not None:
            pred = pred * ignore
            target = target * ignore
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

        batch_size = pred.size(0)
        pred = pred.contiguous().view(batch_size, -1)
        target = target.contiguous().view(batch_size, -1).float()

        a = torch.sum(pred * target, 1)
        b = torch.sum(pred * pred, 1) + 0.001
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)
        dice_loss = 1 - d

        dice_loss = self.loss_weight * dice_loss

        if reduction == 'mean':
            dice_loss = torch.mean(dice_loss)
        return dice_loss
