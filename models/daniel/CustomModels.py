import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import GoogLeNet

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
    return GoogLeNet(num_classes=6, aux_logits=False, init_weights=False)


class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super(SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
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
    def __init__(self, dim: int, amount: int =3):
        super(ConvNeXt_Block, self).__init__()
        self.amount = amount


        self.conv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim)
        self.conv2 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.gelu = nn.GELU()
        self.conv3 = nn.Conv2d(4 * dim, dim, kernel_size=1)

        self.layer_scale = LayerScale(dim)
                    

    def forward(self, x: torch.Tensor):
        for _ in range(self.amount):
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

class EmoNeXt_Tiny(nn.Module):
    def __init__(self):
        super(EmoNeXt_Tiny, self).__init__()
        
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
        self.patch1 = nn.Conv2d(3, 96, kernel_size=4, stride=4) # (64, 64, 3) -> (16, 16, 96)
        self.norm1 = nn.LayerNorm(96)

        self.conv1 = ConvNeXt_Block(96, 3) # (16, 16, 96) -> (16, 16, 96)
        
        self.patch2 = nn.Conv2d(96, 192, kernel_size=2, stride=2) # (16, 16, 96) -> (8, 8, 192)
        self.se1 = SE_Block(192)

        self.conv2 = ConvNeXt_Block(192, 3)

        self.patch3 = nn.Conv2d(192, 384, kernel_size=2, stride=2) # (8, 8, 192) -> (4, 4, 384)
        self.se2 = SE_Block(384)

        self.conv3 = ConvNeXt_Block(384, 9)

        self.patch4 = nn.Conv2d(384, 768, kernel_size=2, stride=2) # (4, 4, 384) -> (2, 2, 768)
        self.se3 = SE_Block(768)

        self.conv4 = ConvNeXt_Block(768, 3)

        self.avg = nn.AvgPool2d(kernel_size=2) # (2, 2, 768) -> (1, 1, 768)
        self.norm2 = nn.LayerNorm(768)

        self.fc = nn.Linear(768, 6)





    def stn(self, x: torch.Tensor):
        B, C, H, W = x.size()

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

