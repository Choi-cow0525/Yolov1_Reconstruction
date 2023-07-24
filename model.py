# start of YOLO reconstruction
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Yolov1(nn.Module):
    def __init__(self, conf) -> None:
        super().__init__()

        self.ConvBlock = ConvLayer(conf)
        self.FCLayer = FCLayer(conf)

    def forward(self, x) -> torch.Tensor:
        grid = self.ConvBlock(x) # 7 x 7 x 1024
        result = self.FCLayer(grid) # 7 x 7 x 30
        
        return result
    
class ConvLayer(nn.Module):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf
        self.conv = nn.Sequential( ## 448 x 448 x 3
            ConvBlock1(conf.conv1), # 112 x 112 x 64
            ConvBlock1(conf.conv2), # 56 x 56 x 192
            ConvBlock2(conf.conv3), # 28 x 28 x 512
            ConvBlock3(conf.conv4), # 14 x 14 x 1024
            ConvBlock4(conf.conv5), # 7 x 7 x 1024
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class ReductConvBlock(nn.Module):
    def __init__(self, in_ch, int_ch, out_ch, ks, strd) -> None:
        super(ReductConvBlock, self).__init__()

        self.reduct = nn.Conv2d(
            in_channels=in_ch,
            out_channels=int_ch,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.conv = nn.Conv2d(
            in_channels=int_ch,
            out_channels=out_ch,
            kernel_size=ks,
            stride=strd,
            padding=(ks-1)//2
        )

        self.norm2_int = nn.BatchNorm2d(int_ch)
        self.norm2_out = nn.BatchNorm2d(out_ch)
    
    def forward(self, x):
        x = self.reduct(x)
        x = F.leaky_relu(self.norm2_int(x), negative_slope=0.1)
        x = self.conv(x)
        x = F.leaky_relu(self.norm2_out(x), negative_slope=0.1)
        return x


class SimpleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, strd) -> None:
        super(ReductConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=ks,
            stride=strd,
            padding=(ks-1)//2
        )

        self.norm2_out = nn.BatchNorm2d(out_ch)
    
    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(self.norm2_out(x), negative_slope=0.1)
        return x


class ConvBlock1(nn.Module):
    def __init__(self, conf) -> None:
        super(ConvBlock1, self).__init__()
        self.conf = conf
        self.in_ch = conf.in_ch
        self.out_ch = conf.out_ch
        self.ks = conf.ks
        self.strd = conf.strd

        self.conv = nn.Conv2d(
            in_channels = self.in_ch,
            out_channels = self.out_ch,
            kernel_size = self.ks,
            stride = self.strd,
            padding = (self.ks-1) // 2)
        
        self.maxpool = nn.MaxPool2d(
            kernel_size = 2, 
            stride = 2)
        
        self.norm2 = nn.BatchNorm2d(self.out_ch)
    
    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        x = F.leaky_relu(self.norm2(x), negative_slope=0.1)
        x = self.maxpool(x)
        return x 
    

class ConvBlock2(nn.Module):
    def __init__(self, conf) -> None:
        super(ConvBlock2, self).__init__()
        self.conf = conf

        self.in_ch = conf.in_ch
        self.int1_ch = conf.int1_ch
        self.int2_ch = conf.int2_ch
        self.out_ch = conf.out_ch
        self.ks = conf.ks
        self.strd = conf.strd

        self.rconv = nn.Sequential(
            ReductConvBlock(in_ch=self.in_ch, int_ch=self.int1_ch, out_ch=self.int2_ch, ks=3, strd=1),
            ReductConvBlock(in_ch=self.int2_ch, int_ch=self.int2_ch, out_ch=self.out_ch, ks=3, strd=1),
        )
        
        self.maxpool = nn.MaxPool2d(
            kernel_size = 2, 
            stride = 2)
        
    def forward(self, x) -> torch.Tensor:
        x = self.rconv(x)
        x = self.maxpool(x)
        

class ConvBlock3(nn.Module):
    def __init__(self, conf) -> None:
        super(ConvBlock3, self).__init__()
        self.conf = conf

        self.in_ch = conf.in_ch
        self.int1_ch = conf.int1_ch
        self.int2_ch = conf.int2_ch
        self.out_ch = conf.out_ch
        self.ks = conf.ks
        self.strd = conf.strd

        self.rconv = nn.Sequential(
            ReductConvBlock(in_ch=self.in_ch, int_ch=self.int1_ch, out_ch=self.int2_ch, ks=3, strd=1),
            ReductConvBlock(in_ch=self.int2_ch, int_ch=self.int1_ch, out_ch=self.int2_ch, ks=3, strd=1),
            ReductConvBlock(in_ch=self.int2_ch, int_ch=self.int1_ch, out_ch=self.int2_ch, ks=3, strd=1),
            ReductConvBlock(in_ch=self.int2_ch, int_ch=self.int1_ch, out_ch=self.int2_ch, ks=3, strd=1),
            ReductConvBlock(in_ch=self.int2_ch, int_ch=self.int2_ch, out_ch=self.out_ch, ks=3, strd=1),
        )

        self.maxpool = nn.MaxPool2d(
            kernel_size = 2, 
            stride = 2)

    def forward(self, x) -> torch.Tensor:
        x = self.rconv(x)
        x = self.maxpool(x)


class ConvBlock4(nn.Module):
    def __init__(self, conf) -> None:
        super(ConvBlock3, self).__init__()
        self.conf = conf
        self.in_ch = conf.in_ch
        self.int_ch = conf.int_ch
        self.out_ch = conf.out_ch
        self.ks = conf.ks

        self.rconv = nn.Sequential(
            ReductConvBlock(in_ch=self.in_ch, int_ch=self.int_ch, out_ch=self.out_ch, ks=3, strd=1),
            ReductConvBlock(in_ch=self.out_ch, int_ch=self.int_ch, out_ch=self.out_ch, ks=3, strd=1),
            SimpleConvBlock(in_ch=self.out_ch, out_ch=self.out_ch, ks=3, strd=1),
            SimpleConvBlock(in_ch=self.out_ch, out_ch=self.out_ch, ks=3, strd=2),
            SimpleConvBlock(in_ch=self.out_ch, out_ch=self.out_ch, ks=3, strd=1),
            SimpleConvBlock(in_ch=self.out_ch, out_ch=self.out_ch, ks=3, strd=1),
        )

    def forward(self, x):
        x = self.rconv(x)


class FCLayer(nn.Module):
    def __init__(self, conf) -> None:
        super(FCLayer, self).__init__()
        self.S = conf.S
        self.B = conf.B
        self.C = conf.C
        self.lin = nn.Sequential(
            nn.Linear(in_features=conf.in_dim, out_features=conf.int_dim),
            nn.BatchNorm1d(conf.int_dim),
            F.leaky_relu(conf.int_dim, negative_slope=0.1),
            nn.Linear(in_features=conf.int_dim, out_features=conf.out_dim),
            nn.BatchNorm1d(conf.out_dim),
            F.sigmoid(),
            nn.Dropout1d(p=0.2),
        )

    def forward(self, x):
        x = nn.Flatten(x)
        out = self.lin(x)
        out = out.reshape(self.S, self.S, self.B * 5 + self.C)
        return out