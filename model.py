import torch
import torch.nn as nn
# image 640 *640 *3

class CNNBlock(nn.Module):
    def __init__(self,in_channel,out_channel,bn_act = True,act = True,**kwargs):
        super().__init__()
        self.bn_act = bn_act
        self.act = act
        self.conv = nn.Conv2d(
            in_channels = in_channel
            ,out_channels = out_channel,**kwargs)
        if self.bn_act:
            self.BN = nn.BatchNorm2d(out_channel)
        self.LeakRelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self,x):
        x = self.conv(x)
        if self.bn_act:
            x = self.BN(x)
        if self.act:
            x = self.LeakRelu(x)
        return x

class ResUnit(nn.Module):
    def __init__(self,in_channel,num_rep):
        super().__init__()
        self.num_rep = num_rep 
        self.in_channel = in_channel
        if in_channel//2 == 0:
            out_channel = 1
        else: 
            out_channel = self.in_channel//2

        self.basic_layers = nn.Sequential(
            CNNBlock(
                in_channel= self.in_channel,
                out_channel = out_channel,
                kernel_size =1,stride =1),
            CNNBlock(
                in_channel= out_channel,
                out_channel = self.in_channel,
                kernel_size =3,stride =1,padding =1), 
        )
        self.layers = nn.ModuleList()
        for _ in range(self.num_rep):
            self.layers += [
                self.basic_layers
            ]
    def forward(self,x):
        for layer in self.layers:
            Residual = x
            x = layer(x)
            x = x + Residual
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes,num_box):
        super().__init__()
        self.repLayer = ResUnit(in_channels,num_rep=1)
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5)*num_box, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes
        self.num_box = num_box

    def forward(self, x):
        x = self.repLayer(x)
        x = self.pred(x) #torch.Size([8, 45, 26, 26])
        x = x.reshape(
            x.shape[0], 
            self.num_box, 
            self.num_classes + 5, 
            x.shape[2], 
            x.shape[3]
        ) #torch.Size([8, 3, 15, 26, 26])
        x = x.permute(0, 1, 3, 4, 2) # torch.Size([8, 3, 26, 26, 15])
        return x

class DarkNet(nn.Module):
    def __init__(self,in_channel,num_classes=10,num_box = 3):
        super().__init__()
        self.inlayers = nn.Sequential(
            CNNBlock(in_channel,32,kernel_size =3,padding = 1),
            CNNBlock(32,64,kernel_size =3,stride = 2,padding = 1),
        )
        self.one_rep_layers = ResUnit(in_channel=64,num_rep=1)
        self.cnn01 = CNNBlock(in_channel=64,out_channel=128,kernel_size =3,stride = 2,padding = 1)
        self.pred1 = ScalePrediction(in_channels=128, num_classes=num_classes,num_box = num_box)
        self.two_rep_layers = ResUnit(in_channel=128,num_rep=2)
        self.cnn02 = CNNBlock(in_channel=128,out_channel=256,kernel_size =3,stride = 2,padding = 1)
        self.pred2 = ScalePrediction(in_channels=256, num_classes=num_classes,num_box = num_box)
        self.third_rep_layers = ResUnit(in_channel=256,num_rep=8)
        self.cnn03 = CNNBlock(in_channel=256,out_channel=512,kernel_size =3,stride = 2,padding = 1)
        self.pred3 = ScalePrediction(in_channels=512, num_classes=num_classes,num_box = num_box)
    def forward(self, x):
        output = []
        x = self.inlayers(x)
        x = self.one_rep_layers(x)
        x = self.cnn01(x)
        output.append(self.pred1(x))
        x = self.two_rep_layers(x)
        x = self.cnn02(x)
        output.append(self.pred2(x))
        x = self.third_rep_layers(x)
        x = self.cnn03(x)
        output.append(self.pred3(x))
        for i in output:
            print(i.shape)
        return output


