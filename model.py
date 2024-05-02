import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.Spectral = nn.utils.spectral_norm(nn.Linear(20, 40))
        self.activation = nn.LeakyReLU(0.2,inplace=True)

    def forward (self, x):
        x= self.conv(x)
        x= self.norm(x)
        x= self.Spectral(x.mean([2,3]))
        x= self.activation(x)
        return x
    
class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1):
        super(Deconv, self).__init__()
        self.deconv= nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.Spectral = nn.utils.spectral_norm(nn.Linear(20, 40))
        self.activation = nn.LeakyReLU(0.2,inplace=True)

    def forward (self, x):
        x= self.conv(x)
        x= self.norm(x)
        x= self.Spectral(x.mean([2,3]))
        x= self.activation(x)
        return x

class MCDC(nn.Module):
    #   Multi-scale Cascaded Dilated Convolution for Image Network
    def __init__(self, in_channels, out_channels):
        super(MCDC, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_channels)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, dilation=3, padding=3)
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, dilation=5, padding=5)
        )
        self.branch4 = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, dilation=3, padding=3),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, dilation=5, padding=5)
        )
        self.conv = ConvBlock(4 * out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv(x)
        return x



class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ConvBlock(256, 512, kernel_size=3, stride=2, padding=1),
            ConvBlock(512, 512, kernel_size=3, stride=2, padding=1),
        )
        self.mcdc = MCDC(512, 512)
        self.decoder = nn.Sequential(
            Deconv(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Deconv(1024, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Deconv(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            Deconv(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(128, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        x = self.mcdc(x)
        skips = list(reversed(skips[:-1]))
        for block, skip in zip(self.decoder[:4], skips):
            x = block(x)
            x = torch.cat([x, skip], dim=1)
        x = self.decoder[4:](x)
        return x

