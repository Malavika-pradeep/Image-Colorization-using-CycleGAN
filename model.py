import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

# MCDC module placeholder (assuming it's a custom block, you can replace it with the actual implementation)
class MCDCModule(nn.Module):
    def __init__(self, in_channels):
        super(MCDCModule, self).__init__()

        # First branch: standard convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        # Second branch: dilated convolution with dilation rate of 3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        # Third branch: dilated convolution with dilation rate of 5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        # Fourth branch: dilated convolution with dilation rate of 7
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=7, dilation=7),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        # Branch 5: Shortcut connection
        self.shortcut = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        shortcut = self.shortcut(x)

        x= out1 + out2 + out3 + out4 + shortcut
        return x


# Define the generator architecture
class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc5 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # MCDC module
        self.mcdc = MCDCModule(512)

        # Decoder
        self.dec1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec3 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec4 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoding path with skip connections
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        enc5_out = self.enc5(enc4_out)

        # Middle module
        mcdc_out = self.mcdc(enc5_out)

        # Decoding path with skip connections
        dec1_out = self.dec1(mcdc_out + enc5_out)
        dec2_out = self.dec2(dec1_out + enc4_out)
        dec3_out = self.dec3(dec2_out + enc3_out)
        dec4_out = self.dec4(dec3_out + enc2_out)
        dec5_out = self.dec5(dec4_out + enc1_out)

        return dec5_out
    



# Self-Attention module
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        out = self.gamma * out + x
        return out

# Define the discriminator architecture
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.self_attention1 = SelfAttention(128)
        self.self_attention2 = SelfAttention(256)

        self.final_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.self_attention1(x)
        x = self.conv3(x)
        x = self.self_attention2(x)
        x = self.conv4(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x


