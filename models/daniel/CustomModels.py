import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import GoogLeNet

EmoNeXt_Architecture = {
        "Tiny": ([96, 192, 384, 768], [3, 3, 9, 3]),
        "Small": ([96, 192, 384, 768], [3, 3, 27, 3]),
        "Base": ([128, 256, 512, 1024], [3, 3, 27, 3]),
        "Large": ([192, 384, 768, 1536], [3, 3, 27, 3]),
        "XLarge": ([256, 512, 1024, 2048], [3, 3, 27, 3]),
}

class VGGNet (nn.Module):

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()

        self.conv1_1 = nn.Conv2d(3, 64, (3, 3), stride=1, padding=1) #(64, 64, 3)
        self.conv1_2 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1) #(64, 64, 64)

        self.pool = nn.MaxPool2d((2, 2), (2, 2)) # (32, 32, 64)

        self.conv2_1 = nn.Conv2d(64, 128, (3, 3), stride=1, padding=1) #(32, 32, 128)
        self.conv2_2 = nn.Conv2d(128, 128, (3, 3), stride=1, padding=1)#(32, 32, 128)

        # pool -> (16, 16, 128)

        self.conv3_1 = nn.Conv2d(128, 256, (3, 3), stride=1, padding=1) #(16, 16, 256)
        self.conv3_2 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1) #(16, 16, 256)
        self.conv3_3 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1) #(16, 16, 256)

        # pool -> (8, 8, 256)

        self.conv4_1 = nn.Conv2d(256, 512, (3, 3), stride=1, padding=1) # (8, 8, 256)
        self.conv4_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1) # (8, 8, 256)
        self.conv4_3 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1) # (8, 8, 256)

        # pool -> (4, 4, 512)

        self.conv5_1 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1) # (4, 4, 512)
        self.conv5_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1) # (4, 4, 512)
        self.conv5_3 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1) # (4, 4, 512)

        # Änderung am originalen Design: Kein pooling da sonst Bild zu klein wird

        self.fc1 = nn.Linear(4 * 4 * 512, 4 * 4 * 256)
        self.fc2 = nn.Linear(4 * 4 * 256, 4 * 4 * 128)
        self.fc3 = nn.Linear(4 * 4 * 128, 6)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))

        x = self.pool(x)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))

        x = self.pool(x)

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))

        x = self.pool(x)

        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))

        x = self.pool(x)

        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))

        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def BuildGoogLeNet(numClasses: int = 6):
    return GoogLeNet(num_classes=numClasses, aux_logits=False, init_weights=False)


class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super(SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.size()
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class LayerScale(nn.Module):
    def __init__(self, channels, init_value=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(channels))

    def forward(self, x):
        return x * self.gamma.view(1, -1, 1, 1)
    
class ConvNeXt_Block(nn.Module):
    def __init__(self, dim: int):
        super(ConvNeXt_Block, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim)
        self.conv2 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.gelu = nn.GELU()
        self.conv3 = nn.Conv2d(4 * dim, dim, kernel_size=1)

        self.layer_scale = LayerScale(dim)
                    

    def forward(self, x: torch.Tensor):
        residual = x

        x = self.conv1(x)

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)

        x = self.layer_scale(x)
        x = x + residual
        
        return x
    
class ConvNeXt_Module(nn.Module):
    def __init__(self, dim: int, amount: int):
        super(ConvNeXt_Module, self).__init__()

        convModules = [ConvNeXt_Block(dim) for _ in range(amount)]
        self.blocks = nn.Sequential(*convModules)


    def forward(self, x):
        self.blocks(x)
        return x
    
class EmoNeXt_Variable(nn.Module):
    def __init__(self, channels: list[int], blocks: list[int]):
        super(EmoNeXt_Variable, self).__init__()

        # Localization network for stn
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(True)
        )

        self.loc_fc = nn.Sequential(
            nn.Linear(12 * 12 * 10, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.loc_fc[2].weight.data.zero_()
        self.loc_fc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

                # Normal Convolutional Layers
        self.patch1 = nn.Conv2d(3, channels[0], kernel_size=4, stride=4) # (64, 64, 3) -> (16, 16, 96)
        self.norm1 = nn.LayerNorm(channels[0])

        self.conv1 = ConvNeXt_Module(channels[0], blocks[0]) # (16, 16, 96) -> (16, 16, 96)
        
        self.patch2 = nn.Conv2d(channels[0], channels[1], kernel_size=2, stride=2) # (16, 16, 96) -> (8, 8, 192)
        self.se1 = SE_Block(channels[1])

        self.conv2 = ConvNeXt_Module(channels[1], blocks[1])

        self.patch3 = nn.Conv2d(channels[1], channels[2], kernel_size=2, stride=2) # (8, 8, 192) -> (4, 4, 384)
        self.se2 = SE_Block(channels[2])

        self.conv3 = ConvNeXt_Module(channels[2], blocks[2])

        self.patch4 = nn.Conv2d(channels[2], channels[3], kernel_size=2, stride=2) # (4, 4, 384) -> (2, 2, 768)
        self.se3 = SE_Block(channels[3])

        self.conv4 = ConvNeXt_Module(channels[3], blocks[3])

        self.avg = nn.AvgPool2d(kernel_size=2) # (2, 2, 768) -> (1, 1, 768)
        self.norm2 = nn.LayerNorm(channels[3])

        self.fc = nn.Linear(channels[3], 6)


    def stn(self, x: torch.Tensor):
        B, _, _, _ = x.size()

        xs = self.localization(x)
        xs = xs.view(B, -1)
        theta = self.loc_fc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x
    
    def forward(self, x: torch.Tensor):
        x = self.stn(x)

        x = self.patch1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.patch2(x)
        x = self.se1(x)

        x = self.conv2(x)
        x = self.patch3(x)
        x = self.se2(x)

        x = self.conv3(x)
        x = self.patch4(x)
        x = self.se3(x)

        x = self.conv4(x)

        x = self.avg(x)
        x = torch.squeeze(x)

        x = self.norm2(x)
        x = self.fc(x)

        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, input: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.inConv =  nn.Sequential(
            nn.Conv2d(channels, input * 4, kernel_size=1,stride=stride),
            nn.BatchNorm2d(input * 4)
            )
        
        self.conv1 = nn.Conv2d(channels, input, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(input)

        self.conv2 = nn.Conv2d(input, input, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(input)

        self.conv3 = nn.Conv2d(input, input * 4, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(input * 4)

        self.relu = nn.ReLU()


    def forward(self, x):
        residual = self.inConv(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.relu(x + residual)

        return x

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3) # (64, 64, 3) -> (32, 32, 64)   Padding added by me so that images does not become too small
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # (32, 32, 64) -> (16, 16, 64)

        self.relu = nn.ReLU()

        self.block1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(256, 64),
            ResidualBlock(256, 64)
        ) # (16, 16, 64) -> (16, 16, 256)

        self.se1 = SE_Block(256)

        self.block2 = nn.Sequential(
            ResidualBlock(256, 128, stride=2),
            ResidualBlock(512, 128),
            ResidualBlock(512, 128)
        ) # (16, 16, 256) -> (8, 8, 512)

        self.se2 = SE_Block(512)

        self.block3 = nn.Sequential(
            ResidualBlock(512, 256, stride=2),
            ResidualBlock(1024, 256),
            ResidualBlock(1024, 256),
            ResidualBlock(1024, 256),
            ResidualBlock(1024, 256),
            ResidualBlock(1024, 256)
        ) # (8, 8, 512) -> (4, 4, 1024)

        self.se3 = SE_Block(1024)

        self.block4 = nn.Sequential(
            ResidualBlock(1024, 512, stride=2),
            ResidualBlock(2048, 512),
            ResidualBlock(2048, 512)
        ) # (4, 4, 1024) -> (2, 2, 2048)

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(2048, 6)
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.se1(x)

        x = self.block2(x)
        x = self.se2(x)

        x = self.block3(x)
        x = self.se3(x)

        x = self.block4(x)

        x = self.pool2(x)

        x = x.squeeze()

        x = self.fc1(x)

        return x
    
class InvertedResidual(nn.Module):
    def __init__(self, inChn: int, outChn: int, exp: int, stride: int = 1):
        super(InvertedResidual, self).__init__()
        expChn = inChn * exp

        self.residual = stride == 1 and inChn == outChn

        self.conv1 = nn.Conv2d(inChn, expChn, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(expChn, expChn, kernel_size=3, padding=1, stride=stride, groups=expChn, bias=False)
        self.conv3 = nn.Conv2d(expChn, outChn, kernel_size=1, bias=False)

        self.relu6 = nn.ReLU6()
        
        self.bn1 = nn.BatchNorm2d(expChn)
        self.bn2 = nn.BatchNorm2d(expChn)
        self.bn3 = nn.BatchNorm2d(outChn)
        

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu6(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu6(y)

        y = self.conv3(y)
        y = self.bn3(y)

        if self.residual:
            y += x
        
        return y

class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.act = nn.SiLU()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False) # (64, 64, 3) -> (32, 32, 32)
        self.bn1 = nn.BatchNorm2d(32)

        self.res1 = InvertedResidual(32, 16, 1, 1) # (32, 32, 32) -> (32, 32, 16)
        self.res2 = nn.Sequential(
            InvertedResidual(16, 24, 6, 2),
            InvertedResidual(24, 24, 6, 1)
        ) # (32, 32, 16) -> (16, 16, 24)
        self.res3 = nn.Sequential(
            InvertedResidual(24, 32, 6, 2),
            InvertedResidual(32, 32, 6, 1),
            InvertedResidual(32, 32, 6, 1)
        ) # (16, 16, 24) -> (8, 8, 32)
        self.res4 = nn.Sequential(
            InvertedResidual(32, 64, 6, 2),
            InvertedResidual(64, 64, 6, 1),
            InvertedResidual(64, 64, 6, 1),
            InvertedResidual(64, 64, 6, 1)
        ) # (8, 8, 32) -> (4, 4, 64)
        self.res5 = nn.Sequential(
            InvertedResidual(64, 96, 6, 1),
            InvertedResidual(96, 96, 6, 1),
            InvertedResidual(96, 96, 6, 1)
        ) # (4, 4, 64) -> (4, 4, 96)
        self.se1 = SE_Block(96)

        self.res6 = nn.Sequential(
            InvertedResidual(96, 160, 6, 1), # Original had stride=2
            InvertedResidual(160, 160, 6, 1),
            InvertedResidual(160, 160, 6, 1)
        ) # (4, 4, 96) -> (4, 4, 160)
        self.se2 = SE_Block(160)

        self.res7 = InvertedResidual(160, 320, 6, 1) # (4, 4, 160) -> (4, 4, 320)
        self.se3 = SE_Block(320)
        
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, bias=False) # (4, 4, 320) -> (4, 4, 1280)
        self.bn2 = nn.BatchNorm2d(1280)

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # (4, 4, 1280) -> (1, 1, 1280)

        self.fc = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 6)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.se1(self.res5(x))
        x = self.se2(self.res6(x))
        x = self.se3(self.res7(x))

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x
        
