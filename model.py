# start of YOLO reconstruction
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Yolov1(nn.Module):
    def __init__(self, conf) -> None:
        super().__init__()

        self.ConvBlock = ConvLayer(conf)
        # self.ConvBlock = Modified_ConvBlock(conf)
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
        super(SimpleConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=ks,
            stride=strd,
            padding=(ks-1)//2,
            bias=False,
        )

        self.norm2_out = nn.BatchNorm2d(out_ch)
    
    def forward(self, x):
        x = F.leaky_relu(self.norm2_out(self.conv(x)), negative_slope=0.1)
        return x


class SConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, strd) -> None:
        super(SConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=ks,
            stride=strd,
            padding=(ks-1)//2,
            bias=False,
        )

        self.norm2_out = nn.BatchNorm2d(out_ch)

        self.lin = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.lin(self.norm2_out(self.conv(x)))
        return x


class ConvBlock1(nn.Module):
    def __init__(self, conf) -> None:
        super(ConvBlock1, self).__init__()
        self.conf = conf
        self.in_ch = conf.in_ch # 3  64
        self.out_ch = conf.out_ch # 64  192
        self.ks = conf.ks # 7  3
        self.strd = conf.strd # 2  1

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
        print("pass1")
        return x 
    

class ConvBlock2(nn.Module):
    def __init__(self, conf) -> None:
        super(ConvBlock2, self).__init__()
        self.conf = conf

        self.in_ch = conf.in_ch # 192
        self.int1_ch = conf.int1_ch # 128
        self.int2_ch = conf.int2_ch # 256
        self.out_ch = conf.out_ch # 512
        self.ks = conf.ks # 3
        self.strd = conf.strd # 1

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
        print("pass2")
        return x
        
class ConvBlock3(nn.Module):
    def __init__(self, conf) -> None:
        super(ConvBlock3, self).__init__()
        self.conf = conf

        self.in_ch = conf.in_ch # 512
        self.int1_ch = conf.int1_ch # 256
        self.int2_ch = conf.int2_ch # 512
        self.out_ch = conf.out_ch # 1024
        self.ks = conf.ks # 3
        self.strd = conf.strd # 1

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
        print("pass3")
        return x


class ConvBlock4(nn.Module):
    def __init__(self, conf) -> None:
        super(ConvBlock4, self).__init__()
        self.conf = conf
        self.in_ch = conf.in_ch # 1024
        self.int_ch = conf.int_ch # 512
        self.out_ch = conf.out_ch # 1024
        self.ks = conf.ks # 3

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
        print("pass4")
        return x


class Modified_ConvBlock(nn.Module):
    def __init__(self, conf):
        super(Modified_ConvBlock, self).__init__()
        self.conf = conf

        self.conv = nn.Sequential(
            SConvBlock(conf.conv1.in_ch, conf.conv1.out_ch, conf.conv1.ks, conf.conv1.strd),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
            SConvBlock(conf.conv2.in_ch, conf.conv2.out_ch, conf.conv2.ks, conf.conv2.strd),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
            SConvBlock(conf.conv3.in_ch, conf.conv3.int1_ch, 1, 1),
            SConvBlock(conf.conv3.int1_ch, conf.conv3.int2_ch, conf.conv3.ks, conf.conv3.strd),
            SConvBlock(conf.conv3.int2_ch, conf.conv3.int2_ch, 1, 1),
            SConvBlock(conf.conv3.int2_ch, conf.conv3.out_ch, conf.conv3.ks, conf.conv3.strd),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
            SConvBlock(conf.conv4.in_ch, conf.conv4.int1_ch, 1, 1),
            SConvBlock(conf.conv4.int1_ch, conf.conv4.int2_ch, conf.conv4.ks, conf.conv4.strd),
            SConvBlock(conf.conv4.int2_ch, conf.conv4.int1_ch, 1, 1),
            SConvBlock(conf.conv4.int1_ch, conf.conv4.int2_ch, conf.conv4.ks, conf.conv4.strd),
            SConvBlock(conf.conv4.int2_ch, conf.conv4.int1_ch, 1, 1),
            SConvBlock(conf.conv4.int1_ch, conf.conv4.int2_ch, conf.conv4.ks, conf.conv4.strd),
            SConvBlock(conf.conv4.int2_ch, conf.conv4.int1_ch, 1, 1),
            SConvBlock(conf.conv4.int1_ch, conf.conv4.int2_ch, conf.conv4.ks, conf.conv4.strd),
            SConvBlock(conf.conv4.int2_ch, conf.conv4.int2_ch, 1, 1),
            SConvBlock(conf.conv4.int2_ch, conf.conv4.out_ch, conf.conv4.ks, conf.conv4.strd),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
            SConvBlock(conf.conv5.in_ch, conf.conv5.int_ch, 1, 1),
            SConvBlock(conf.conv5.int_ch, conf.conv5.out_ch, conf.conv5.ks, conf.conv5.strd),
            SConvBlock(conf.conv5.out_ch, conf.conv5.int_ch, 1, 1),
            SConvBlock(conf.conv5.int_ch, conf.conv5.out_ch, conf.conv5.ks, conf.conv5.strd),
            SConvBlock(conf.conv5.out_ch, conf.conv5.out_ch, ks=3, strd=1),
            SConvBlock(conf.conv5.out_ch, conf.conv5.out_ch, ks=3, strd=2),
            SConvBlock(conf.conv5.out_ch, conf.conv5.out_ch, ks=3, strd=1),
            SConvBlock(conf.conv5.out_ch, conf.conv5.out_ch, ks=3, strd=1),
        )
        print(self.conv)

    def forward(self, x):
        x = self.conv(x)
        return x


class FCLayer(nn.Module):
    def __init__(self, conf) -> None:
        super(FCLayer, self).__init__()
        self.conf = conf
        self.S = conf.grid.S
        self.B = conf.grid.B
        self.C = conf.grid.C
        self.lin1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=conf.linear.in_dim, out_features=conf.linear.int_dim),
            nn.BatchNorm1d(conf.linear.int_dim)
        )
        self.lin2 = nn.Sequential(
            nn.Linear(in_features=conf.linear.int_dim, out_features=conf.linear.out_dim),
            nn.BatchNorm1d(conf.linear.out_dim),
        )
        self.drop = nn.Dropout1d(p=0.5)

    def forward(self, x):
        # print(x.shape)
        x = self.lin1(x)
        # print(x.shape)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.drop(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        x = x.reshape(x.shape[0], self.S, self.S, self.B * 5 + self.C)
        print(f"shape after forward : {x.shape}\n")
        return x