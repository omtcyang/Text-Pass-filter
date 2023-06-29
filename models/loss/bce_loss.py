import torch
import torch.nn as nn


class BalanceCrossEntropyLoss(nn.Module):
    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                mask: torch.Tensor,
                weight_map: torch.Tensor):
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(torch.sigmoid(pred), gt.float(), reduction='none')
        positive_loss = loss * positive.float() * weight_map
        negative_loss = loss * negative.float() * weight_map
        negative_loss, _ = negative_loss.view(-1).topk(negative_count)
        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)
        return balance_loss
