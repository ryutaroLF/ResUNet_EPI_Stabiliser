import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import os

def save_featuremap(tensor,save_directory_path,name):

    if not os.path.exists(save_directory_path):
        os.makedirs(save_directory_path)

    numpy_array_for_img = tensor.squeeze().detach().cpu().numpy()
    numpy_array = np.expand_dims(numpy_array_for_img, axis=0) #(70,512,512)→(1,7,512,512)

    if os.path.exists(os.path.join(save_directory_path,f"{name}.npy")):
        base_numpy_array = np.load(os.path.join(save_directory_path,f"{name}.npy"))
        numpy_array = np.concatenate((base_numpy_array, numpy_array), axis=0)

    batch = numpy_array.shape[0]
    np.save(os.path.join(save_directory_path,name), numpy_array)

    fig, axes = plt.subplots(7, 10, figsize=(20, 14))

    for i, ax in enumerate(axes.flat):
        if i < len(numpy_array_for_img):
            ax.imshow(numpy_array_for_img[i], cmap='gray')
        ax.axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(os.path.join(save_directory_path, f'{batch}_{name}.png'), bbox_inches='tight')
    plt.close(fig)

def save_featuremap_280(tensor,save_directory_path,name,batch):

    if not os.path.exists(save_directory_path):
        os.makedirs(save_directory_path)

    numpy_array_for_img = tensor.squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(28, 10, figsize=(20, 14))

    for i, ax in enumerate(axes.flat):
        if i < len(numpy_array_for_img):
            ax.imshow(numpy_array_for_img[i], cmap='gray')
        ax.axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(os.path.join(save_directory_path, f'{batch}_{name}.png'), bbox_inches='tight')
    plt.close(fig)

def save_single_tensor(tensor,save_directory_path,name):

    numpy_array_for_img = tensor.squeeze().detach().cpu().numpy() #(1,1,490,490)→(490,490)
    numpy_array = np.expand_dims(numpy_array_for_img, axis=0) #(490,490)→(1,490,490)

    if os.path.exists(os.path.join(save_directory_path,f"{name}.npy")):
        base_numpy_array = np.load(os.path.join(save_directory_path,f"{name}.npy"))
        numpy_array = np.concatenate((base_numpy_array, numpy_array), axis=0)

    np.save(os.path.join(save_directory_path,name), numpy_array)

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

class EPINET(nn.Module):

    def __init__(self, input_ch, filter_num, stream_num):
        super(EPINET, self).__init__()

        filter_concatamated = filter_num * stream_num

        self.multistream1 = multi_stream(in_ch = input_ch, out_ch = filter_num)
        self.multistream2 = multi_stream(in_ch = filter_num, out_ch = filter_num)
        self.multistream3 = multi_stream(in_ch = filter_num, out_ch = filter_num)
        self.concatanated_conv1 = concatanated_conv(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.concatanated_conv2 = concatanated_conv(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.concatanated_conv3 = concatanated_conv(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.concatanated_conv4 = concatanated_conv(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.concatanated_conv5 = concatanated_conv(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.concatanated_conv6 = concatanated_conv(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.concatanated_conv7 = concatanated_conv(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.last_conv1 = last_conv(in_ch = filter_concatamated, out_ch = 1)

    def forward(self, x_0d, x_90d, x_45d, x_m45d, batch):
        x_0d_1 = self.multistream1(x_0d)
        x_0d_2 = self.multistream2(x_0d_1)
        x_0d_3 = self.multistream3(x_0d_2)

        x_90d_1 = self.multistream1(x_90d)
        x_90d_2 = self.multistream2(x_90d_1)
        x_90d_3 = self.multistream3(x_90d_2)

        x_45d_1 = self.multistream1(x_45d)
        x_45d_2 = self.multistream2(x_45d_1)
        x_45d_3 = self.multistream3(x_45d_2)

        x_m45d_1 = self.multistream1(x_m45d)
        x_m45d_2 = self.multistream2(x_m45d_1)
        x_m45d_3 = self.multistream3(x_m45d_2)

        x = torch.cat((x_90d_3, x_0d_3, x_45d_3, x_m45d_3), dim=1)

        x_1 = self.concatanated_conv1(x)
        x_2 = self.concatanated_conv2(x_1)
        x_3 = self.concatanated_conv3(x_2)
        x_4 = self.concatanated_conv4(x_3)
        x_5 = self.concatanated_conv5(x_4)
        x_6 = self.concatanated_conv6(x_5)
        x_7 = self.concatanated_conv7(x_6)

        x_8 = self.last_conv1(x_7)

        return x_0d_1,x_0d_2,x_0d_3,x,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8
