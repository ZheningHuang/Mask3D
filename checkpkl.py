import pickle
import numpy as np
import shutil
import os


with open("['scene0011_00']_torch.Size([164833, 28])_pair.pkl", 'rb') as f:
    # Deserialize the tuple and load it into memory
    loaded_tuple = pickle.load(f)

Folder_mask = "matcher_check/scene0011_00/mask_proposal/"
Folder_gt = "matcher_check/scene0011_00/target_proposal/"

FINAL_FOLDER = "matcher_check/scene0011_00/grouped/"

if not os.path.exists(FINAL_FOLDER):
        os.makedirs(FINAL_FOLDER)
        
total_gt_mask = len(loaded_tuple[0][1])

for i in range(total_gt_mask):
    mask_idx = loaded_tuple[0][0][i]
    gt_idx = loaded_tuple[0][1][i]

    mask_name = "mask_proposal_{}.ply".format(mask_idx)
    gt_name = "target_{}.ply".format(gt_idx)
    
    src_mask = Folder_mask + mask_name
    src_gt = Folder_gt + gt_name

    dst_mask = FINAL_FOLDER + "pair_{}_mask_{}.ply".format(i, mask_idx)
    dst_gt = FINAL_FOLDER + "pair_{}_gt_{}.ply".format(i, gt_idx)
    
    shutil.copyfile(src_mask, dst_mask)
    shutil.copyfile(src_gt, dst_gt)