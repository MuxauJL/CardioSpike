import torch.nn as nn
import torch


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
