import torch.nn as nn
import torch


class WeightedBCEWithLogits(nn.BCEWithLogitsLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean', pos_weight=None):
        pos_weight = torch.tensor(pos_weight)
        super().__init__(weight, size_average, reduce, reduction, pos_weight)

