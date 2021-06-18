import torch
import torch.nn as nn


class BatchApply(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x, mask):
        out = []
        length = mask.sum([-1, -2]).int().to(x.device)
        for i in range(x.shape[0]):
            stats = self.func(x[i:i + 1][:, :, :length[i]])
            out.append(stats)
        return torch.cat(out, dim=0)


class MedianPool(nn.Module):
    def forward(self, x):
        return torch.median(x, dim=-1, keepdim=True)[0]


class StdPool(nn.Module):
    def forward(self, x):
        return torch.std(x, dim=[0, -1], keepdim=True)


class StackApply(nn.Module):
    CHANNELS_MULTIPLIER = 4

    def __init__(self, in_features, out_features, inner_layers=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        stats = [nn.AdaptiveMaxPool1d(1), nn.AdaptiveAvgPool1d(1), StdPool(), MedianPool()]
        self.stats = nn.ModuleList([BatchApply(stat) for stat in stats])
        self.stats_downsample = nn.ModuleList([nn.Conv1d(self.in_features, inner_layers, 1)  for stat in stats])


        self.out_layer = nn.Conv1d(self.in_features + (inner_layers * self.CHANNELS_MULTIPLIER), self.out_features, 1)

    def forward(self, x, mask):
        stats_output = []
        for stat, stat_conv in zip(self.stats, self.stats_downsample):
            stat_out = stat_conv(stat(x, mask))
            stats_output.append(stat_out)
        stats_output = torch.cat(stats_output, dim=1)
        stats_output = stats_output.repeat(1, 1, x.shape[2])
        out = torch.cat([x, stats_output], dim=1)
        out = self.out_layer(out)
        return out * mask

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
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.norm1 = LayerNorm(out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.norm2 = LayerNorm(out_channels)
        self.act2 = nn.GELU()
        self.additional_features = StackApply(out_channels, out_channels)

    def forward(self, x, mask):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = out * mask
        save_out = out
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.additional_features(out * mask, mask)
        out = out + save_out

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
        self.channels = channels
        self.hidden_layers = nn.ModuleList()

        for in_c, out_c in zip(channels[:-1], channels[1:]):
            self.hidden_layers.append(SimpleLayerCNN(in_c, out_c))
        if self.output_channels == self.channels[-1]:
            self.last_conv = lambda x: x
        else:
            self.last_conv = nn.Conv1d(channels[-1], self.output_channels, 5, padding=2)


    def forward(self, x, x_diff, time, angle, mask):
        x = torch.cat([x, time, x_diff, angle], dim=2)
        x = x.transpose(2, 1)
        mask = mask.unsqueeze(1)
        for module in self.hidden_layers:
            x = module(x, mask)
        x = self.last_conv(x)
        x = x.transpose(2, 1)
        return x


class CRNN(nn.Module):
    def __init__(self, input_channels=2, output_channels=1, num_convs=10, start_channels=32, multiplier=1.3,
                 lstm_hidden_dim=339, lstm_layers_count=1, out_channels=339):
        super().__init__()

        self.cnn = SimpleCNN(input_channels=input_channels,
                             num_convs=num_convs,
                             output_channels=out_channels,
                             start_channels=start_channels,
                             multiplier=multiplier)
        self.bi_lstm = nn.LSTM(input_size=out_channels, hidden_size=lstm_hidden_dim,
                               num_layers=lstm_layers_count,
                               proj_size=0, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(out_channels + 2 * lstm_hidden_dim, output_channels)

    def forward(self,  x, x_diff, time, angle, mask):
        output = self.cnn(x, x_diff, time, angle, mask)
        features = output
        output, (_, _) = self.bi_lstm(features)
        output = torch.cat((features, output), 2)
        output = self.fc(output)

        return output