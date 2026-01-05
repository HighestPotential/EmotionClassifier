import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Deep_Emotion(nn.Module):
    def __init__(self):
        '''
        Deep_Emotion class contains the network architecture.
        '''
        super(Deep_Emotion,self).__init__()
        self.conv1 = nn.Conv2d(3,10,3)
        self.conv2 = nn.Conv2d(10,10,3)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,10,3)
        self.pool4 = nn.MaxPool2d(2,2)

        self.norm = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(810,50)
        self.fc2 = nn.Linear(50,7)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self,input):
        out = self.stn(input)

        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = F.relu(self.pool2(out))

        out = F.relu(self.conv3(out))
        out = self.norm(self.conv4(out))
        out = F.relu(self.pool4(out))

        out = F.dropout(out)
        out = out.view(-1, 810)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out