import torch
import registry.registries as registry
from models.losses import WeightedBCEWithLogits
from models.simple_cnn import SimpleCNN

registry.Criterion(WeightedBCEWithLogits)

registry.Model(SimpleCNN)

# Generator
def define_model(opt_net):
    which_model = opt_net['model']
    net = registry.MODELS.get_from_params(**opt_net)
    return net
