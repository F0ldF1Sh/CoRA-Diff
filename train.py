"""
Training script for IMDER
dataset_name: Selecting dataset (mosi or mosei)
seeds: This is a list containing running seeds you input
mr: missing rate ranging from 0.1 to 0.7
"""
import os
# set CUDA_VISIBLE_DEVICES before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from run import IMDER_run

seed=42
mr=0.7
dataset_name='mosi'
IMDER_run(model_name='imder',
           dataset_name=dataset_name,
           seeds=[seed],
           mr=mr,
           model_save_dir=f"/root/FoldFish/dataset/CoraDiff-Dataset/result/{dataset_name}/mr{mr}_seed{seed}",
           res_save_dir=f"/root/FoldFish/dataset/CoraDiff-Dataset/result/{dataset_name}/mr{mr}_seed{seed}",
           log_dir=f"/root/FoldFish/dataset/CoraDiff-Dataset/log/{dataset_name}/mr{mr}_seed{seed}",
           resume_from_checkpoint=False)