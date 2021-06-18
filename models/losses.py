import torch.nn as nn
import torch
import torch.nn.functional as F

class WeightedBCEWithLogits(nn.BCEWithLogitsLoss):
    def __init__(self, weight=None, clip=None, size_average=None, reduce=None, reduction: str = 'mean', pos_weight=None):
        pos_weight = torch.tensor(pos_weight)
        self.clip_min = None if clip is None or clip == 0.0 else torch.logit(torch.tensor(clip))
        self.clip_max = None if clip is None or clip == 0.0 else torch.logit(torch.tensor(1 - clip))
        super().__init__(weight, size_average, reduce, reduction, pos_weight)

    def forward(self, input, target):
        if self.clip_min is not None:
            input = torch.clip(input, self.clip_min, self.clip_max)
        return super().forward(input, target)



class LocalBetterLoss(nn.Module):
    def forward(self, predictions, targets, mask):
        positives_count = torch.sum(targets, dim=1) + 1
        negatives_count = mask.sum(1).unsqueeze(1) - torch.sum(targets, dim=1)

        pos_weight = negatives_count / positives_count
        pos_weight = pos_weight.unsqueeze(2).repeat(1, targets.shape[1], 1)
        loss = F.binary_cross_entropy_with_logits(predictions[mask], targets[mask], pos_weight=pos_weight[mask])

        return loss
