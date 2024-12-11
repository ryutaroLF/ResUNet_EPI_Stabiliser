import sys
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
if current_directory not in sys.path:
    sys.path.append(current_directory)

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from mobilevit import *

#############################################
#               multi_stream
#############################################
class multi_stream(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(multi_stream, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 2, padding=0),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 2, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):

        x = self.conv(x)
        return x

"""
MV2Block(in_ch, out_ch, kernel=2, stride=1, padding=False, expansion=4)

in_ch : MbileNet v2ブロックへの入力チャンネル次元数
out_ch : MbileNet v2ブロックの出力チャンネル次元数
kernel : DepthWise-Convolutionのカーネル
stride : DepthWise-Convolutionのストライド
padding : DepthWise-Convolutionのパディング
expansion : Point-Wise Convolutionの拡大率
"""

"""
MobileViTBlock(dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.)

dim : Transformerの共通次元
depth : Transformerブロックの数
channel : 入力次元．
kernel_size : 畳み込みのカーネルサイズ
patch_size : パッチ画像のサイズ
mlp_dim : FFNの隠れ層のサイズ
dropout : ドロップアウトの割合
"""


class multi_stream_ViT(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=2, stride=1, padding=0, expansion=8, skip_con=False):
        super(multi_stream_ViT, self).__init__()
        """
        self.conv = nn.Sequential(
            MV2Block(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, expansion=expansion, skip_con=skip_con),
            MV2Block(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, expansion=expansion, skip_con=skip_con),
            MobileViTBlock(dim=256, depth=2, channel=out_ch, kernel_size=2, patch_size=(2,2), mlp_dim=256, dropout=0.)
        )
        """
        self.conv = nn.Sequential(
            MV2Block(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, expansion=expansion, skip_con=skip_con),
            MV2Block(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, expansion=expansion, skip_con=skip_con)
        )

    def forward(self, x):

        #print(f"multi_stream  INPUT size {x.size()}")
        x = self.conv(x)
        #print(f"multi_stream  OUTPUT size {x.size()}")

        return x


class multi_stream_DWSC(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=2, stride=1, padding=0, expansion=8, skip_con=False):
        super(multi_stream_DWSC, self).__init__()

        self.conv = nn.Sequential(
            DWSCBlock(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, skip_con=skip_con)
        )

    def forward(self, x):

        x = self.conv(x)

        return x
    
#############################################
#               concatanated_conv
#############################################

class concatanated_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(concatanated_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 2, padding=0),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 2, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):

        x = self.conv(x)
        return x


class concatanated_conv_ViT(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=2, stride=1, padding=0, expansion=4, skip_con=False):
        super(concatanated_conv_ViT, self).__init__()
        """
        self.conv = nn.Sequential(
            MV2Block(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, expansion=expansion, skip_con=skip_con),
            MV2Block(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, expansion=expansion, skip_con=skip_con),
            MobileViTBlock(dim=512, depth=4, channel=out_ch, kernel_size=2, patch_size=(2,2), mlp_dim=512, dropout=0.)
        )
        """
        self.conv = nn.Sequential(
            MV2Block(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, expansion=expansion, skip_con=skip_con),
            MV2Block(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, expansion=expansion, skip_con=skip_con)
        )


    def forward(self, x):

        x = self.conv(x)
        return x


class concatanated_conv_DWSC(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=2, stride=1, padding=0, expansion=4, skip_con=False):
        super(concatanated_conv_DWSC, self).__init__()

        self.conv = nn.Sequential(
            DWSCBlock(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding,  skip_con=skip_con)
        )


    def forward(self, x):

        x = self.conv(x)
        return x
#############################################
#               last_conv
#############################################

class last_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(last_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 2, padding=0),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(in_ch, out_ch, 2, padding=0),
        )

    def forward(self, x):

        x = self.conv(x)
        return x


class last_conv_ViT(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=2, stride=1, padding=0, expansion=4, skip_con=False):
        super(last_conv_ViT, self).__init__()
        """
        self.conv = nn.Sequential(
            MV2Block(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, expansion=expansion, skip_con=skip_con),
            MV2Block(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, expansion=expansion, skip_con=skip_con),
            MobileViTBlock(dim=512, depth=2, channel=out_ch, kernel_size=2, patch_size=(2,2), mlp_dim=512, dropout=0.)
        )
        """
        self.conv = nn.Sequential(
            MV2Block(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, expansion=expansion, skip_con=skip_con),
            MV2Block(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, expansion=expansion, skip_con=skip_con)
        )
    def forward(self, x):

        x = self.conv(x)
        return x
    
class last_conv_DWSC(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=2, stride=1, padding=0, expansion=4, skip_con=False):
        super(last_conv_DWSC, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, 2, padding=0)
        )
    def forward(self, x):

        x = self.conv(x)
        return x
    
#############################################
#               Main
#############################################


class EPINET_D(nn.Module):

    def __init__(self, input_ch, filter_num, stream_num):
        super(EPINET_D, self).__init__()

        filter_concatamated = filter_num * stream_num

        self.multistream1 = multi_stream_DWSC(in_ch = input_ch, out_ch = filter_num)
        self.multistream2 = multi_stream_DWSC(in_ch = filter_num, out_ch = filter_num)
        self.multistream3 = multi_stream_DWSC(in_ch = filter_num, out_ch = filter_num)
        self.concatanated_conv1 = concatanated_conv_DWSC(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.concatanated_conv2 = concatanated_conv_DWSC(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.concatanated_conv3 = concatanated_conv_DWSC(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.concatanated_conv4 = concatanated_conv_DWSC(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.concatanated_conv5 = concatanated_conv_DWSC(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.concatanated_conv6 = concatanated_conv_DWSC(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.concatanated_conv7 = concatanated_conv_DWSC(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.last_conv1 = last_conv_DWSC(in_ch = filter_concatamated, out_ch = 1)

    def forward(self, x):
        x = self.multistream1(x)
        x = self.multistream2(x)
        x = self.multistream3(x)

        x = self.concatanated_conv1(x)
        x = self.concatanated_conv2(x)
        x = self.concatanated_conv3(x)
        x = self.concatanated_conv4(x)
        x = self.concatanated_conv5(x)
        x = self.concatanated_conv6(x)
        x = self.concatanated_conv7(x)

        x = self.last_conv1(x)

        return x
