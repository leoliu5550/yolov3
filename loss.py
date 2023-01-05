import torch
import torch.nn as nn


class Yololoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self,target):
        # 第一層判斷背景與否的矩陣
        # 二元矩陣
        self.obj = target[..., 0] == 1
        self.noobj = target[..., 0] == 0

        return None

lossfunc = Yololoss()
target = torch.zeros(8,3,26,26,15)
target[0,0,0,0,0]=1

lossfunc(target)
