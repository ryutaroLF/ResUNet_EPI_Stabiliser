

import os
import sys

import torch
import numpy as np
import torch.nn as nn
from torchvision.models import vgg16
import torch.nn.functional as F

from network.model_for_eval_simple import *
from network.model import *

import matplotlib.pyplot as plt


def read_pfm(fpath, expected_identifier="Pf"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html
    
    def _get_next_line(f):
        next_line = f.readline().decode('utf-8').rstrip()
        # ignore comments
        while next_line.startswith('#'):
            next_line = f.readline().rstrip()
        return next_line
    
    with open(fpath, 'rb') as f:
        #  header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise Exception('Could not parse dimensions: "%s". '
                            'Expected "width height", e.g. "512 512".' % line_dimensions)

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
        except:
            raise Exception('Could not parse max value / endianess information: "%s". '
                            'Should be a non-zero number.' % line_scale)

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))
            data = np.flipud(data)
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

        return data
    


def B9HW_to_BH9W_EPI_gen(input_tensor, batchsize):
    if batchsize != input_tensor.size(0):
        print(f"Error : check B9HW_to_BH9W_EPI_gen")

    B = input_tensor.size(0)
    H = input_tensor.size(2)
    W = input_tensor.size(3)

    result_tensor = input_tensor.permute(0, 2, 1, 3).reshape(B, H, 9, W)

    return result_tensor



def save_model_and_optimizer(model, optimizer, save_path):

    directory_path = os.path.dirname(save_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    state = {
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    }

    torch.save(state, save_path)
    






class VGG16Features(nn.Module):
    def __init__(self, layers=['relu1_2']):
        super(VGG16Features, self).__init__()
        self.vgg16 = vgg16(pretrained=True).features
        self.layers = layers

    def forward(self, x):
        features = []
        for name, layer in self.vgg16._modules.items():
            x = layer(x)
            if name in self.layers:
                features.append(x)
                break
        return features[0]

def VGG16_loss(tensor1, tensor2):

    # tensor 1 ->(1, 256, 9, 256)
    # tensor 2 ->(1, 256, 9, 256)

    segments1 = tensor1.chunk(4, dim=1)
    segments2 = tensor2.chunk(4, dim=1)

    vgg_features = VGG16Features(layers=['4']).cuda()  # relu1_2は4番目のレイヤー

    mse_values = []

    for seg1, seg2 in zip(segments1, segments2):
        reshaped_segment1 = seg1.reshape(BATCHSIZE, -1, seg1.size(3))  # (1, 64*9, 256)
        reshaped_segment2 = seg2.reshape(BATCHSIZE, -1, seg2.size(3))  # (1, 64*9, 256)
        rgb_segment1 = reshaped_segment1.unsqueeze(1).repeat(1, 3, 1, 1).cuda()  # (16, 3, 576, 256)
        rgb_segment2 = reshaped_segment2.unsqueeze(1).repeat(1, 3, 1, 1).cuda()  # (16, 3, 576, 256)
      
        with torch.no_grad():
            features1 = vgg_features(rgb_segment1)
            features2 = vgg_features(rgb_segment2)

        mse = F.mse_loss(features1, features2)
        mse_values.append(mse.item())

    average_mse = sum(mse_values) / len(mse_values)

    return average_mse

def MSFR(pred, target, scales=[1, 0.5, 0.25]):
    loss = 0.0
    for scale in scales:
        # リサイズ
        scaled_pred = F.interpolate(pred, scale_factor=scale, mode='nearest')
        scaled_target = F.interpolate(target, scale_factor=scale, mode='nearest')
        
        # 周波数領域に変換
        pred_freq = torch.fft.fft2(scaled_pred)
        target_freq = torch.fft.fft2(scaled_target)
        
        # L1損失を計算
        loss += F.l1_loss(pred_freq, target_freq)
    return loss / len(scales)


def multi_scale_l1_loss(pred, target, scales=[1, 0.5, 0.25]):
    loss = 0.0
    for scale in scales:
        scaled_pred = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_target = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
        loss += F.l1_loss(scaled_pred, scaled_target)
    return loss / len(scales)



class CustomLoss:
    def __init__(self, scales=[1, 0.5, 0.25]):
        self.scales = scales

    def __call__(self, pred, target):
        msfr_loss = MSFR(pred, target, scales=self.scales)
        l1_loss = multi_scale_l1_loss(pred, target, scales=self.scales)
        return msfr_loss + l1_loss