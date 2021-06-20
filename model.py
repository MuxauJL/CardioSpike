import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Construct a layer-norm for 1d features."""

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
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.norm1 = LayerNorm(out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.norm2 = LayerNorm(out_channels)
        self.act2 = nn.GELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        save_out = out
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = out + save_out
        return out


class SimpleCNN(nn.Module):
    def __init__(self, input_channels=2, num_convs=10, start_channels=32, multiplier=1.3):
        super().__init__()
        self.multiplier = multiplier
        self.start_channels = start_channels
        self.num_convs = num_convs
        self.input_channels = input_channels

        channels = [self.input_channels, *[round(self.start_channels * multiplier ** i) for i in range(self.num_convs)]]
        self.output_channels = channels[-1]

        self.hidden_layers = nn.ModuleList()

        for in_c, out_c in zip(channels[:-1], channels[1:]):
            self.hidden_layers.append(SimpleLayerCNN(in_c, out_c))

    def freeze_pretrained_layers(self, freeze=True):
        for layer in self.hidden_layers[1:-round(self.num_convs / 3 * 2)]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        for module in self.hidden_layers:
            x = module(x)
        return x


class CRNN(nn.Module):
    def __init__(self, input_channels=2, num_convs=10, start_channels=32, multiplier=1.3,
                 lstm_hidden_dim=256, lstm_layers_count=1):
        super().__init__()

        self.cnn = SimpleCNN(input_channels=input_channels,
                             num_convs=num_convs,
                             start_channels=start_channels,
                             multiplier=multiplier)
        self.bi_lstm = nn.LSTM(input_size=self.cnn.output_channels, hidden_size=lstm_hidden_dim,
                               num_layers=lstm_layers_count,
                               proj_size=0, bidirectional=True)
        self.fc = nn.Linear(self.cnn.output_channels + 2 * lstm_hidden_dim, 1)

    def forward(self, inputs):
        output = self.cnn(inputs)
        features = output.permute(2, 0, 1)
        output, (_, _) = self.bi_lstm(features)
        output = torch.cat((features, output), 2)
        output = self.fc(output)
        return output
