import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod


class BaseSegLoader(Dataset, ABC):
    
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
    
    @abstractmethod
    def __len__(self):
    
        pass
    
    @abstractmethod
    def __getitem__(self, index):
       
        pass



class SWaTSegLoader(BaseSegLoader):

    def __init__(self, data_path, win_size, step, mode="train"):
        super().__init__(data_path, win_size, step, mode)
        raise NotImplementedError("Implementation is not included in the open-source version.")
    
    def __len__(self):
        raise NotImplementedError("Implementation is not included in the open-source version.")
    
    def __getitem__(self, index):
        raise NotImplementedError("Implementation is not included in the open-source version.")


class PSMSegLoader(BaseSegLoader):
  
    def __init__(self, data_path, win_size, step, mode="train"):
        super().__init__(data_path, win_size, step, mode)
        raise NotImplementedError("Implementation is not included in the open-source version.")


class MSLSegLoader(BaseSegLoader):
   
    def __init__(self, data_path, win_size, step, mode="train"):
        super().__init__(data_path, win_size, step, mode)
        raise NotImplementedError("Implementation is not included in the open-source version.")


class SMAPSegLoader(BaseSegLoader):
 
    def __init__(self, data_path, win_size, step, mode="train"):
        super().__init__(data_path, win_size, step, mode)
        raise NotImplementedError("Implementation is not included in the open-source version.")


class SMDSegLoader(BaseSegLoader):

    def __init__(self, data_path, win_size, step, mode="train"):
        super().__init__(data_path, win_size, step, mode)
        raise NotImplementedError("Implementation is not included in the open-source version.")






def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD', val_ratio=0.2):
 
    dataset_map = {
        'SMD': SMDSegLoader,
        'MSL': MSLSegLoader,
        'SMAP': SMAPSegLoader,
        'PSM': PSMSegLoader,
        'SWaT': SWaTSegLoader,
   
    }
    
    if dataset not in dataset_map:
        raise ValueError(f"Unsupported dataset type: {dataset}")
    

    dataset_instance = dataset_map[dataset](data_path, win_size, step, mode)
    

    shuffle = False
    if mode == 'train':
        shuffle = True

    
    return DataLoader(dataset=dataset_instance, batch_size=batch_size, shuffle=shuffle)