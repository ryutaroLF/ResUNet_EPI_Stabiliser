
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Adjust the input channels if necessary
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)

class ResUNet_DWSC(nn.Module):
    def __init__(self, in_channels, out_channels, filter_num):
        super(ResUNet_DWSC, self).__init__()
        self.encoder1 = self.conv_block(in_channels, filter_num)
        self.encoder2 = self.conv_block(filter_num, filter_num*2)
        self.encoder3 = self.conv_block(filter_num*2, filter_num*4)
        self.encoder4 = self.conv_block(filter_num*4, filter_num*8)
        self.bottleneck = self.conv_block(filter_num*8, filter_num*16)
        self.decoder4 = self.conv_block(filter_num*16 + filter_num*8, filter_num*8)
        self.decoder3 = self.conv_block(filter_num*8 + filter_num*4, filter_num*4)
        self.decoder2 = self.conv_block(filter_num*4 + filter_num*2, filter_num*2)
        self.decoder1 = self.conv_block(filter_num*2 + filter_num, filter_num)
        self.final_conv = nn.Conv2d(filter_num, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        dec4 = self.decoder4(torch.cat((F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True), enc4), dim=1))
        dec3 = self.decoder3(torch.cat((F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True), enc3), dim=1))
        dec2 = self.decoder2(torch.cat((F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True), enc2), dim=1))
        dec1 = self.decoder1(torch.cat((F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True), enc1), dim=1))
        return self.final_conv(dec1)