
import os
import sys
sys.path.append(os.path.abspath(r"F:\lab\EPINET_MobileViT\EPI_Stabilization\UNet_EPI_Stabilization\epinet_fun"))

from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import random
from torchvision.models import vgg16
import torchvision.transforms as transforms

from network.model_for_eval_simple import *
from network.model import *

from config.config_v2_1 import *

class CustomDataset:
    def __init__(self, base_path, dataset_list,flag_print_distorted_path, flag_GT_output=False):
        self.base_path = base_path
        self.dataset_list = dataset_list
        self.dataset_len = len(dataset_list)
        self.n_steps = 8
        self.mu = 0
        self.sigma = 1
        self.top_idx = list(range(0, 9))
        self.bottom_idx = list(range(72, 81))
        self.flag_print_distorted_path = flag_print_distorted_path
        self.flag_GT_output = flag_GT_output

    def __getitem__(self, index):

        random_dataset_idx = random.randint(0, self.dataset_len-1)
        dataset_path = []
        dataset_path = os.path.join(self.base_path, self.dataset_list[random_dataset_idx])
        current_position = 36
        positions = [current_position]

        ########################ランダムな多視点画像を作成########################
        # ガウス分布のパラメータを設定して次の位置を計算
        for _ in range(self.n_steps):
            next_position = self.get_next_position(current_position)
            positions.append(next_position)
            current_position = next_position
        
        # ランダムな位置の画像
        images = []
        transform = transforms.ToTensor()
        for pos in positions:
            img_name = f"input_Cam{pos:03d}.png"
            img_path = os.path.join(dataset_path, img_name)
            img = Image.open(img_path).convert('RGB')
            img = img.convert('L')  # 次にモノクロに変換する
            img_tensor = transform(img)
            images.append(img_tensor)

        # テンソルを積み重ねて[i, H, W]の次元にする
        stacked_images = torch.stack(images)
        crop_size = (CROP_SIZE, CROP_SIZE)
        distorted_multiview_tensor, top, left = self.random_crop(stacked_images, crop_size)
        distorted_multiview_tensor = distorted_multiview_tensor.squeeze()
        if self.flag_print_distorted_path is True: print(f"positions: {positions}")
        
        ########################ランダムな多視点画像を作成########################

        # 正しい多視点画像
        images = []
        transform = transforms.ToTensor()
        positions = [36,37,38,39,40,41,42,43,44]
        for pos in positions:
            img_name = f"input_Cam{pos:03d}.png"
            img_path = os.path.join(dataset_path, img_name)
            img = Image.open(img_path).convert('RGB')
            img = img.convert('L')  # 次にモノクロに変換する
            img_tensor = transform(img)
            images.append(img_tensor)

        # テンソルを積み重ねて[i, H, W]の次元にする
        stacked_images = torch.stack(images)
        crop_size = (CROP_SIZE, CROP_SIZE)
        not_distorted_multiview_tensor = self.normal_crop(stacked_images, crop_size,top,left)
        not_distorted_multiview_tensor = not_distorted_multiview_tensor.squeeze()

        if self.flag_GT_output is True:
            pfm_path = os.path.join(dataset_path, "gt_disp_lowres.pfm")
            GT_tensor = torch.from_numpy(read_pfm(pfm_path).copy())
            return distorted_multiview_tensor, not_distorted_multiview_tensor, GT_tensor

        else:
            return distorted_multiview_tensor, not_distorted_multiview_tensor

    def __len__(self):
        # データセットの長さを返す。ここでは仮の値を返している。
        return BATCHSIZE * NUMof1epoch

    def get_next_position(self, pos):
        displacement = np.random.normal(self.mu, self.sigma)

        if pos in self.top_idx:
            next_pos = pos + 1
            if displacement > 1:
                next_pos = pos + 1
            elif displacement > 0.5:
                next_pos = pos + 10  # 右斜め下
            elif displacement < -1:
                next_pos = pos + 10  # 右斜め下
            else:
                next_pos = pos + 1
            
        elif pos in self.bottom_idx:
            next_pos = pos + 1
            if displacement > 1:
                next_pos = pos + 1
            elif displacement > 0.5:
                next_pos = pos - 8  # 右斜め上
            elif displacement < -1:
                next_pos = pos - 8  # 右斜め上
            else:
                next_pos = pos + 1

        else:
            next_pos = pos + 1
            if displacement > 1:
                next_pos = pos + 1
            elif displacement > 0.5:
                next_pos = pos - 8  # 右斜め上
            elif displacement < -1:
                next_pos = pos + 10  # 右斜め下
            else:
                next_pos = pos + 1

        return next_pos
    
    def random_crop(self,tensor, crop_size):
        _, _, height, width = tensor.shape
        crop_height, crop_width = crop_size

        # ランダムな開始位置を決定
        top = torch.randint(0, height - crop_height + 1, (1,)).item()
        left = torch.randint(0, width - crop_width + 1, (1,)).item()

        # 26x26のサイズを切り出し
        cropped_tensor = tensor[:, :, top:top + crop_height, left:left + crop_width]
        return cropped_tensor, top, left

    def normal_crop(self,tensor, crop_size, top, left):
        _, _, height, width = tensor.shape
        crop_height, crop_width = crop_size

        # 26x26のサイズを切り出し
        cropped_tensor = tensor[:, :, top:top + crop_height, left:left + crop_width]
        return cropped_tensor
    