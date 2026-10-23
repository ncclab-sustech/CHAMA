import os
from tkinter import Y
# from scipy.io import loadmat, savemat
import pandas as pd
import scipy.io as scio
import torch
import numpy as np
from torch.utils.data import Dataset


class Input_Data(Dataset):
    def __init__(self, data, gt_dyn, trial_id, mask,training_mask,condition_idx=None,beh_data=None, gt_mask=None):
        super().__init__()
        self.data = data
        self.gt_dyn = gt_dyn
        self.trial_id = trial_id
        self.mask = mask
        self.training_mask = training_mask
        self.condition_idx = condition_idx
        self.beh_data = beh_data
        self.gt_mask = gt_mask
    def __len__(self):

        return self.data.shape[0]

    def __getitem__(self, idx):

        segment_data = self.data[idx, :, :]
        if self.gt_dyn is not None: 
            segment_gt_dyn = self.gt_dyn[idx,:,:]
        segment_trial_id = self.trial_id[idx]
        segment_mask = self.mask[idx,:]
        segment_training_mask = self.training_mask[idx,:]
        if self.condition_idx is not None:
            segment_condition_idx = self.condition_idx[idx]
            if self.beh_data is not None:
                segment_beh_data = self.beh_data[idx]
                if self.gt_mask is not None:
                    segment_gt_mask = self.gt_mask[idx]
                    return segment_data,segment_trial_id,segment_condition_idx,segment_beh_data,segment_mask,segment_training_mask,segment_gt_mask
                else:
                    return segment_data,segment_trial_id,segment_condition_idx,segment_beh_data,segment_mask,segment_training_mask
            else:
                return segment_data,segment_trial_id,segment_condition_idx,segment_mask,segment_training_mask

        return segment_data,segment_gt_dyn,segment_trial_id,segment_mask,segment_training_mask

