import os
import sys

import torch
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from network.model_for_eval_simple import *
from network.model import *


from config.config_v4_4 import *
from utils.utils import *
from utils.datasets import *
from utils.model import *
from utils.model2 import *

def main():
    print("Start Initialization...")
    base_path = "../../dataset/hci_dataset"
    train_dataset = CustomDataset(base_path, dataset_list=train_dataset_list, flag_print_distorted_path=False)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=48)

    print("Initialized Train Dataset & Dataloader")
    val_dataset = CustomDataset(base_path, dataset_list=val_dataset_list, flag_print_distorted_path=False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=48)
    print("Initialized Val Dataset & Dataloader")
    
    UNet = ResUNet_DWSC(in_channels=9, out_channels=9,filter_num=256).cuda()
    print("Created Models")

    optimizer = optim.Adam(UNet.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.5)
    l1_loss = nn.L1Loss()
    best_validation_loss = 100
    criterion = CustomLoss()

    print(f"Start training & validation loop!")
    for epoch in range(MAX_EPOCH):
        train_itr = 0
        val_itr = 0

        loss_train_total = 0
        loss_train_full = 0
        loss_train_EPI  = 0
        loss_train_vgg  = 0
        loss_val_total = 0
        loss_val_full = 0
        loss_val_EPI  = 0
        loss_val_vgg  = 0

        UNet.train()


        for data in train_dataloader:
            
            input = data[0].cuda()
            GT = data[1].cuda()
            optimizer.zero_grad()
            
            output = UNet(input)

            loss_total = criterion(GT, output)
            loss_total.backward()
            optimizer.step()

            loss_train_total += loss_total.item()
            train_itr += 1

        for data in val_dataloader:
            UNet.eval()
            with torch.no_grad():
                input = data[0].cuda()
                GT = data[1].cuda()            
                output = UNet(input)

                loss_total = criterion(GT, output)
                loss_val_total += loss_total.item()
                val_itr += 1

        log = f"Epoch [{epoch+1}/{MAX_EPOCH}], Train total :{loss_train_total/train_itr:.4f} Val total: {loss_val_total/val_itr:.4f} "
        print(log)
        with open(f"results_v4_4.txt", 'a') as file:
            file.write(log)


        loss_comp = loss_val_total/val_itr
        if best_validation_loss > loss_comp:
            train_loss_str = f"{loss_train_total/train_itr:.4f}".replace(".","p")
            val_loss_str = f"{loss_val_total/val_itr:.4f}".replace(".","p")
            file_name = f"epoch{epoch}_TRloss{train_loss_str}_VALloss{val_loss_str}"
            save_model_and_optimizer(UNet, optimizer, rf"checkpoints/v4_4/{file_name}.pth")

            best_validation_loss = loss_comp

        scheduler.step()
    print("Training completed.")


if __name__ == "__main__":
    main()