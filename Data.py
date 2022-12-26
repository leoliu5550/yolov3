import os
import torch 
from torch.utils.data import Dataset
# 640 × 640
class ChessDataset(Dataset):
    def __init__(self,subpath):
        # DATA/train
        self.datatype = ['images','labels']
        self.subpath = subpath 
        self.image_path = os.path.join(self.subpath,self.datatype[0])
        self.label_path = os.path.join(self.subpath,self.datatype[1])

    def __len__(self):
        return len(os.listdir(self.image_path))

    def __getitem__(self, index):
        return torch.set
        