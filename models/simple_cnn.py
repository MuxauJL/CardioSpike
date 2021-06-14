import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    "Construct a layernorm for 1d features."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(1, features, 1))
        self.b_2 = nn.Parameter(torch.zeros(1, features, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SimpleLayerCNN(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.norm1 = LayerNorm(out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.norm2 = LayerNorm(out_channels)
        self.act2 = nn.GELU()

    def forward(self, x, mask):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = out * mask
        save_out = out
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = (out * mask) + save_out
        return out


class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_channels, num_convs=6, start_channels=32, multiplier=1.2):
        super().__init__()
        self.multiplier = multiplier
        self.start_channels = start_channels
        self.num_convs = num_convs
        self.output_channels = output_channels
        self.input_channels = input_channels + 1 + 1 + 1  # + x_diff + time + angle

        channels = [self.input_channels, *[round(self.start_channels * multiplier ** i) for i in range(self.num_convs)]]

        self.hidden_layers = nn.ModuleList()

        for in_c, out_c in zip(channels[:-1], channels[1:]):
            self.hidden_layers.append(SimpleLayerCNN(in_c, out_c))

        self.last_conv = nn.Conv1d(channels[-1], self.output_channels, 5, padding=2)

    def freeze_pretrained_layers(self, freeze=True):
        for layer in self.hidden_layers[1:-round(self.num_convs / 3 * 2)]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x, x_diff, time, angle, mask):
        x = torch.cat([x, time, x_diff, angle], dim=2)
        x = x.transpose(2, 1)
        mask = mask.unsqueeze(1)
        for module in self.hidden_layers:
            x = module(x, mask)
        x = self.last_conv(x)
        x = x.transpose(2, 1)
        return x
