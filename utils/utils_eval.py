import numpy as np
import os 
import imageio
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


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



def save_validation_tensor_as_png(tensor,save_path):

    directory_path = os.path.dirname(save_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    tensor = tensor.detach().cpu().numpy()
    normalized_image = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    image_uint8 = np.uint8(normalized_image * 255)

    concatenated_images = np.hstack(image_uint8)
    imageio.imsave(save_path, np.squeeze(concatenated_images))

def save_tensor_as_png(tensor,save_path):

    tensor = tensor.detach().cpu().numpy()
    imageio.imsave(save_path, np.squeeze(tensor))
    


class UNet_with_BN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_num):
        super(UNet_with_BN, self).__init__()
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
    
    def conv_block(self, in_channels, out_channels, dropout_prob=0.5):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)
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
    
def make_distorted_positions():

    current_position = 36
    positions = [current_position]

    for _ in range(8):
        next_position = get_next_position(current_position)
        positions.append(next_position)
        current_position = next_position
    return positions

def make_distorted_EPI_GT(positions, dataset_path, CROP_SIZE):
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
    distorted_multiview_tensor = torch.stack(images)
    crop_size = (CROP_SIZE, CROP_SIZE)
    distorted_multiview_tensor_crop, top, left = random_crop(distorted_multiview_tensor, crop_size)
    distorted_multiview_tensor_crop = distorted_multiview_tensor_crop.squeeze()
    
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
    not_distorted_multiview_tensor = torch.stack(images)
    crop_size = (CROP_SIZE, CROP_SIZE)
    not_distorted_multiview_tensor_crop = normal_crop(not_distorted_multiview_tensor, crop_size,top,left)
    not_distorted_multiview_tensor_crop = not_distorted_multiview_tensor_crop.squeeze()

    pfm_path = os.path.join(dataset_path, "gt_disp_lowres.pfm")
    GT_tensor = torch.from_numpy(read_pfm(pfm_path).copy())
    return distorted_multiview_tensor.squeeze(1), not_distorted_multiview_tensor.squeeze(1), distorted_multiview_tensor_crop, not_distorted_multiview_tensor_crop, GT_tensor


def get_next_position(pos):
    displacement = np.random.normal(0, 1)

    top_idx = list(range(0, 9))
    bottom_idx = list(range(72, 81))

    if pos in top_idx:
        next_pos = pos + 1
        if displacement > 1:
            next_pos = pos + 1
        elif displacement > 0.5:
            next_pos = pos + 10  # 右斜め下
        elif displacement < -1:
            next_pos = pos + 10  # 右斜め下
        else:
            next_pos = pos + 1
        
    elif pos in bottom_idx:
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



def random_crop(tensor, crop_size):
    _, _, height, width = tensor.shape
    crop_height, crop_width = crop_size

    # ランダムな開始位置を決定
    top = torch.randint(0, height - crop_height + 1, (1,)).item()
    left = torch.randint(0, width - crop_width + 1, (1,)).item()

    # 26x26のサイズを切り出し
    cropped_tensor = tensor[:, :, top:top + crop_height, left:left + crop_width]
    return cropped_tensor, top, left

def normal_crop(tensor, crop_size, top, left):
    _, _, height, width = tensor.shape
    crop_height, crop_width = crop_size

    # 26x26のサイズを切り出し
    cropped_tensor = tensor[:, :, top:top + crop_height, left:left + crop_width]
    return cropped_tensor


def save_EPI_comparison(distorted_multiview, restored_multiview, not_distorted_multiview, file_name='EPI_comparison.png'):
    input_EPI = B9HW_to_BH9W_EPI_gen(distorted_multiview.unsqueeze(0), 1)
    GT_EPI = B9HW_to_BH9W_EPI_gen(restored_multiview.unsqueeze(0),1)
    output_EPI = B9HW_to_BH9W_EPI_gen(not_distorted_multiview.unsqueeze(0), 1)
    
    img1 = input_EPI[0,10, :, :].numpy()
    img2 = output_EPI[0,10, :, :].numpy()
    img3 = GT_EPI[0,10,:,:].numpy()

    # 2つの画像を上下に並べてサブプロット
    fig, axes = plt.subplots(3, 1, figsize=(5, 4))  # figsizeを調整して見やすくする

    # 最初の画像をプロット
    axes[0].imshow(img1, cmap='gray', aspect='auto')
    axes[0].set_title('EPI of the Distorted Multiview', loc='left', pad=20)  # タイトルを左に配置
    axes[0].axis('off')  # 軸を非表示にする

    # 2番目の画像をプロット
    axes[1].imshow(img2, cmap='gray', aspect='auto')
    axes[1].set_title('EPI of the Restored Multiview', loc='left', pad=20)  # タイトルを左に配置
    axes[1].axis('off')  # 軸を非表示にする

    # 3番目の画像をプロット
    axes[2].imshow(img3, cmap='gray', aspect='auto')
    axes[2].set_title('EPI of the Ground-Truth Multiview', loc='left', pad=20)  # タイトルを左に配置
    axes[2].axis('off')  # 軸を非表示にする

    plt.tight_layout()  # プロットが重ならないようにレイアウトを調整

    # 画像として保存
    plt.savefig(file_name, bbox_inches='tight', dpi=300)  # 指定されたファイル名で保存
    plt.close()


def calculate_metrics(img1, img2):
    # RMSEの計算
    mse_value = np.mean((img1 - img2) ** 2)
    rmse_value = np.sqrt(mse_value)
    
    # 画像のデータ範囲を取得
    data_range = img2.max() - img2.min()
    
    # SSIMの計算
    ssim_value = ssim(img1, img2, data_range=data_range)
    
    # PSNRの計算
    psnr_value = psnr(img1, img2, data_range=data_range)
    
    return rmse_value, ssim_value, psnr_value