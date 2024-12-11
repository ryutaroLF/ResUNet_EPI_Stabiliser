#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=06:00:00
#$ -N LF_v2_2
#$ -o log/output_v2_2.log
#$ -e log/error_v2_2.log

echo "Job started"
date

echo "Checking disk space..."
df -h

# Change cache directory
export TORCH_HOME=/gs/fs/tga-nn/cache

# Load modules
module load cuda/12.1.0
module load cudnn/9.0.0
module load nccl/2.20.5

echo "Basic modules loaded"

# Create virtual environment
#python3 -m venv EPI_UNet

# Activate virtual environment
source EPI_UNet/bin/activate
python3 -m pip install --upgrade pip

# Install PyTorch
pip install torch==2.3.0 torchvision==0.18.0 --trusted-host download.pytorch.org --index-url https://download.pytorch.org/whl/cu121
echo "Installed torch and torchvision using pip3"
echo "PyTorch version:"
python -c "import torch; print('PyTorch Version:', torch.__version__)"
export TORCH_HOME=/gs/fs/tga-nn/cache

# Install additional packages
echo "Start installing all packages:"
pip3 install scikit-learn imbalanced-learn torchvision tqdm matplotlib pandas numpy xarray einops h5py

# Execute Python script
echo "Running Python script"
torchrun --nproc_per_node=1 main_v2_2_MSFR_loss.py
