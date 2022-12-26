import pytest
import sys
sys.path.append("//Users//leoliu//Documents//yolov3_torch")

from model import *

class TestCNNBlock:
    def test_withoutBN(self):
        model = CNNBlock(
            in_channel= 3,
            out_channel=32,
            kernel_size =3,
            stride = 1,
            bn_act = False,
            padding = 1
        )
        x = torch.rand(1,3,320,320)
        assert model(x).shape == torch.Size([1,32,320,320])
    def test_withBN(self):
        model = CNNBlock(
            in_channel= 3,
            out_channel=32,
            kernel_size =3,
            stride = 1,
            bn_act = True,
            padding = 1
        )
        x = torch.rand(1,3,320,320)
        assert model(x).shape == torch.Size([1,32,320,320])
    def test_withoutAct(self):
        model = CNNBlock(
            in_channel= 3,
            out_channel=32,
            kernel_size =3,
            stride = 1,
            bn_act = True,
            act = False,
            padding = 1
        )
        x = torch.rand(1,3,320,320)
        assert model(x).shape == torch.Size([1,32,320,320])

class TestResUnit:
    def test_ResUnit(self):
        model = ResUnit(
            in_channel=4,num_rep=8
        )
        x = torch.ones(1,4,320,320)
        assert model(x).shape == torch.Size([1,4,320,320])

@pytest.mark.skip(reason="don't need currently test this")
class TestDarkNet:
    @pytest.mark.skip(reason="don't need currently test this")
    def test_DarkNet01(self): 
        model = DarkNet(in_channel=3)
        x= torch.ones(8,3,416,416)
        assert model(x)[0].shape == torch.Size([8,64,208,208])
        assert model(x)[1].shape == torch.Size([8,64,208,208])

    @pytest.mark.skip(reason="don't need currently test this")
    def test_DarkNet02(self):
        model = DarkNet(in_channel=3)
        x= torch.ones(8,3,416,416)
        assert model(x)[2].shape == torch.Size([8,128,104,104])
        assert model(x)[3].shape == torch.Size([8,128,104,104])
        
    @pytest.mark.skip(reason="don't need currently test this")
    def test_DarkNet03(self):
        model = DarkNet(in_channel=3)
        x= torch.ones(8,3,416,416)
        assert model(x)[4].shape == torch.Size([8,256,52,52])
        assert model(x)[5].shape == torch.Size([8,256,52,52])

    def test_DarkNet04(self):
        model = DarkNet(in_channel=3)
        x= torch.ones(8,3,416,416)
        assert model(x).shape == torch.Size([8,512,26,26])

class TestScalePrediction:
    def test_ScalePred1(self):
        model = ScalePrediction(in_channels=512,num_classes=10,num_box = 3)
        x= torch.ones(8,512,26,26)
        assert model(x).shape == torch.Size([8,3,26,26,15])

class TestDarkNetOutput:
    def test_Output01(self): 
        model = DarkNet(in_channel=3)
        x= torch.ones(8,3,416,416)
        assert model(x)[0].shape == torch.Size([8, 3, 104, 104, 15])

    def test_Output02(self): 
        model = DarkNet(in_channel=3)
        x= torch.ones(8,3,416,416)
        assert model(x)[1].shape == torch.Size([8, 3, 52, 52, 15])

    def test_Output03(self): 
        model = DarkNet(in_channel=3)
        x= torch.ones(8,3,416,416)
        assert model(x)[2].shape == torch.Size([8, 3, 26, 26, 15])
