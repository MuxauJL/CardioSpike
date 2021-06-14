""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = LayerNorm(mid_channels)
        self.act1 =nn.GELU()
        self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = LayerNorm(out_channels)
        self.act2 = nn.GELU()


    def forward(self, x, mask):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = out * mask
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = out * mask
        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.max = nn.MaxPool1d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, mask):
        x = self.max(x)
        mask = self.max(mask)
        return self.conv(x, mask)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2, mask):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, mask)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, start_channels=32, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels + 3, start_channels)
        self.down1 = Down(start_channels, start_channels * 2**1)
        self.down2 = Down(start_channels * 2**1, start_channels * 2**2 // factor)
        #self.down3 = Down(start_channels * 2**2, start_channels * 2**3 )
        #self.down4 = Down(start_channels * 2**3, start_channels * 2**4 // factor)
        #self.up1 = Up(start_channels * 2**4, start_channels * 2**3 // factor, bilinear)
        #self.up2 = Up(start_channels * 2**3, start_channels * 2**2 // factor, bilinear)
        self.up3 = Up(start_channels * 2**2, start_channels * 2**1 // factor, bilinear)
        self.up4 = Up(start_channels * 2**1, start_channels, bilinear)
        self.outc = OutConv(start_channels, n_classes)

    def forward(self, x, x_diff, time, angle, mask):
        x = torch.cat([x, time, x_diff, angle], dim=2)
        x = x.transpose(2, 1)
        mask = mask.unsqueeze(1)
        N,C,H = mask.shape
        x1 = self.inc(x, mask)
        x2 = self.down1(x1, F.adaptive_max_pool1d(mask, output_size=H))
        x3 = self.down2(x2, F.adaptive_max_pool1d(mask, output_size=H//2))
        #x4 = self.down3(x3, F.adaptive_max_pool1d(mask, output_size=H//4))
        #x5 = self.down4(x4, F.adaptive_max_pool1d(mask, output_size=H//8))
        #x = self.up1(x5, x4, F.adaptive_max_pool1d(mask, output_size=H//8))
        #x = self.up2(x4, x3, F.adaptive_max_pool1d(mask, output_size=H//4))
        x = self.up3(x3, x2, F.adaptive_max_pool1d(mask, output_size=H//2))
        x = self.up4(x, x1, F.adaptive_max_pool1d(mask, output_size=H))
        logits = self.outc(x) * mask
        logits = logits.transpose(2, 1)
        return logits

    # def freeze_pretrained_layers(self, freeze=True):
    #     hidden_layers = [self.down1, self.down2,  self.down3, self.down4, self.up1, self.up2]
    #     for layer in hidden_layers:
    #         for param in layer.parameters():
    #             param.requires_grad = not freeze