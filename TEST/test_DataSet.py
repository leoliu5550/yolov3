import sys,os
sys.path.append("//Users//leoliu//Documents//yolov3_torch")
import pytest
from Data import *
from model import *

class TestChessDataGeneral:
    path = [
        r'DATA/train/images',
        r'DATA/train/labels',
        r'DATA/test/images',
        r'DATA/test/labels',
        r'DATA/valid/images',
        r'DATA/valid/labels']
    numfile = [9,9,1,1,2,2]
    def testDataLen(self):
        for num,p in zip(self.numfile,self.path):
            assert num == len(os.listdir(p))

    def testChessDataset__len__(self):
        path = 'DATA/train'
        DATA = ChessDataset(path)
        assert DATA.__len__() == 9
    
class TestChessData:
    path = 'DATA/train'
    DATA = ChessDataset(path)
    def testData__init(self):
        pass

    def testAttribute(self):
        pass

    def testChessDataset__len__(self):
        pass
        