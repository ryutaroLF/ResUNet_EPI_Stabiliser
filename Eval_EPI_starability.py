import torch
from torch.utils.data import DataLoader
from utils.utils_eval import *
from network.model_for_eval_simple import *
from network.model import *

train_dataset_list = [
            "additional/antinous", "additional/boardgames", "additional/dishes",   "additional/greek",
            "additional/kitchen",  "additional/medieval2",  "additional/museum",   "additional/pens",
            "additional/pillows",  "additional/platonic",   "additional/rosemary", "additional/table",
            "additional/tomb",     "additional/tower",      "additional/town",     "additional/vinyl" ]

val_dataset_list = [
            "stratified/backgammon", "stratified/dots", "stratified/pyramids", "stratified/stripes",
            "training/boxes", "training/cotton", "training/dino", "training/sideboard"]

base_path = r"F:\lab\EPINET_MobileViT\dataset\hci_dataset"
UNet = UNet_with_BN(in_channels=9, out_channels=9, filter_num=64).cuda()
checkpoint = torch.load(r"C:\checkpoints\v2_3\epoch431_TRloss0p7341_VALloss0p6980.pth")
UNet.load_state_dict(checkpoint['model_state'])
UNet.eval()

fig, axes = plt.subplots(3, len(val_dataset_list), figsize=(24, 6))
labels = ["Distorted", "Restored", "Not-Distorted"]
for i, label in enumerate(labels):
    fig.text(0.04, 0.85 - i * 0.3, label, ha='center', va='center', rotation=0, size='large')

# generate positions
positions = make_distorted_positions()
print(f"Distorted positions : {positions}")

for dataset_idx in range(len(val_dataset_list)):

    dataset_path = os.path.join(base_path,val_dataset_list[dataset_idx])
    distorted_multiview, not_distorted_multiview, distorted_multiview_crop, not_distorted_multiview_crop, GT = make_distorted_EPI_GT(positions, dataset_path, 128)

    with torch.no_grad():
        restored_multiview = UNet(distorted_multiview.unsqueeze(0).cuda()).cpu().squeeze(0)

    save_EPI_comparison(distorted_multiview, restored_multiview, not_distorted_multiview, file_name=f'EPI_comparison_{dataset_idx}.png')

    input_size=23+2         # Input size should be greater than or equal to 23
    label_size=input_size-22 # Since label_size should be greater than or equal to 1
    Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])  # number of views ( 0~8 for 9x9 )

    EPINETD = EPINET_D(input_ch = 9, filter_num = 70, stream_num =1).to("cuda")
    checkpoint = torch.load(r"F:\lab\EPINET_MobileViT\EPINET_Torch\1Stream_DWSC_v1\model_checkpoint\epoch_194882_training_Loss_updated_loss_0p024426649082452057.pth")
    EPINETD.load_state_dict(checkpoint['model_state'])
    EPINETD.eval()

     
    depth_distorted = EPINETD(distorted_multiview.unsqueeze(0).float().clone().to("cuda"))
    depth_distorted_np = depth_distorted.cpu().detach().numpy()[0, 0, :, :]

    depth_restorted = EPINETD(restored_multiview.unsqueeze(0).float().clone().to("cuda"))
    depth_restorted_np = depth_restorted.cpu().detach().numpy()[0, 0, :, :]

    depth_not_distorted = EPINETD(not_distorted_multiview.unsqueeze(0).float().clone().to("cuda"))
    depth_not_distorted_np = depth_not_distorted.cpu().detach().numpy()[0, 0, :, :]

    rmse1, ssim1, psnr1 = calculate_metrics(depth_distorted_np, GT[11:501, 11:501].numpy())
    rmse2, ssim2, psnr2 = calculate_metrics(depth_restorted_np, GT[11:501, 11:501].numpy())
    rmse3, ssim3, psnr3 = calculate_metrics(depth_not_distorted_np, GT[11:501, 11:501].numpy())


    # 最初の画像とメトリクスをプロット
    axes[0,dataset_idx].imshow(depth_distorted_np, cmap='gray')
    axes[0,dataset_idx].axis('off')
    axes[0,dataset_idx].set_title(f'RMSE: {rmse1:.4f}\n SSIM: {ssim1:.4f}\n PSNR: {psnr1:.2f} dB', fontsize=10)

    # 2番目の画像とメトリクスをプロット
    axes[1,dataset_idx].imshow(depth_restorted_np, cmap='gray')
    axes[1,dataset_idx].axis('off')
    axes[1,dataset_idx].set_title(f'RMSE: {rmse2:.4f}\n SSIM: {ssim2:.4f}\n PSNR: {psnr2:.2f} dB', fontsize=10)

    # 3番目の画像とメトリクスをプロット
    axes[2,dataset_idx].imshow(depth_not_distorted_np, cmap='gray')
    axes[2,dataset_idx].axis('off')
    axes[2,dataset_idx].set_title(f'RMSE: {rmse3:.4f}\n SSIM: {ssim3:.4f}\n PSNR: {psnr3:.2f} dB', fontsize=10)

plt.tight_layout()  # レイアウトの調整
plt.subplots_adjust(left=0.02)
plt.show()
