import torch
import torch.nn as nn 
import torch.nn.functional as F

from DMSHN import DMSHN

class stacked_DMSHN(nn.Module):
    def __init__(self):
        super(stacked_DMSHN,self).__init__()
        self.net1 = DMSHN()
        self.net2 = DMSHN()

    def forward(self,x):
        out1 = self.net1(x)
        out2 = self.net2(out1)

        return out2
