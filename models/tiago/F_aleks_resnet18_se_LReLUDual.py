import torch
import torch.nn as nn

class SEBlock(nn.Module): 
    def __init__(self, channels, reduction=16): 
        super().__init__()
        hidden = max(1, channels // reduction)

        self.pool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
class BasicBlockSE(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, reduction=16):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch) 

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.se = SEBlock(out_ch, reduction=reduction)
        self.leakyrelu = nn.LeakyReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x

        out = self.leakyrelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.leakyrelu(out + identity)
        return out
    
class ResNet18SE(nn.Module):
    def __init__(self, num_classes=6, reduction=16):
        super().__init__()
        self.in_ch = 64

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.leakyrelu = nn.LeakyReLU(inplace=True)

        self.layer1 = self._make_layer(64, 2, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(128, 2, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(256, 2, stride=2, reduction=reduction)
        self.layer4 = self._make_layer(512, 2, stride=2, reduction=reduction)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 *2, num_classes)

    def _make_layer(self, out_ch, blocks, stride, reduction):
        layers = [BasicBlockSE(self.in_ch, out_ch, stride=stride, reduction=reduction)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlockSE(self.in_ch, out_ch, stride=1, reduction=reduction))
        return nn.Sequential(*layers)
    
    def extract_features(self, x):
        x = self.leakyrelu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.flatten(1)
    
    def forward(self, x1,x2):
        f1 = self.extract_features(x1)
        f2 = self.extract_features(x2)

        x = torch.cat([f1, f2], dim=1)  # [B, 1024]
        return self.fc(x)