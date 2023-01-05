import sys,os
sys.path.append("//Users//leoliu//Documents//yolov3_torch")
import pytest
from loss import *

    
class Testloss:
    def testcase(self):
        lossfunc = Yololoss()
        target = torch.ones(8,3,26,26,15)
        obj = lossfunc(target).obj
        noobj = lossfunc(target).noobj
        assert obj.shape == torch.Size([8,3,26,26])
        assert noobj.shape == torch.Size([8,3,26,26])
