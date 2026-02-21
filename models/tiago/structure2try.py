"""
Docstring for structure

"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch.optim import Adam
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import Counter


from pathlib import Path

# class eca_layer

class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
    """
    def __init__(self, channel, k_size = 5):
        super(ECA, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(1,1, kernel_size=k_size, padding= (k_size - 1)// 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
       # x: input features with shape [b,c,h,w]

       # Features descriptor on the global spatial information
       y = self.avg_pool(x)

       # Two different branches of ECA module
       y = self.conv(y.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)

       # Multi-scale information fusion
       y= self.sigmoid(y)

       return x * y.expand_as(x)




# class  MFM
class MFM(nn.Module):
    def __init__(self, in_channels, out_channels, type=1):
        super().__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=3, padding=1)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        x1, x2 = torch.split(x, self.out_channels, dim=1)
        return torch.max(x1, x2)





#class MAake_layer 
    # conv
    # batch norm
    # mfm
class ConvMFMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_layers=1, kernel_size=3, stride=1, padding=1):
        super().__init__()
        layers = []
        for i in range(conv_layers):
            conv_in = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(conv_in, 2*out_channels, kernel_size, stride, padding))
            layers.append(nn.BatchNorm2d(2*out_channels))  # BatchNorm before MFM
            layers.append(MFM(in_channels=2*out_channels, out_channels=out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

#trying to make it fit, now 5 conv
class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.Sequential(
            ConvMFMBlock(3, 48, conv_layers=1),   # 2 convs
            ConvMFMBlock(48, 96, conv_layers=2),  # 3 convs
            ConvMFMBlock(96, 192, conv_layers=2), # 3 convs
            ConvMFMBlock(192, 256, conv_layers=2) # 4 convs
            )

        def forward(self,x):
            x = self.blocks(x)
            x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
            return x


#class CERN with eca, patches

class CERN(nn.Module):
    def __init__(self, num_classes=6, num_regions=4):
        super().__init__()

        self.backbone = Backbone()
        self.num_regions = num_regions
    
        self.eca = nn.ModuleList([ECA(256,3) for i in range(num_regions+1)])  # REVER

        self.pool = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in range(num_regions+1)])


        self.region_net = nn.ModuleList([ nn.Sequential( nn.Linear(256,256), nn.ReLU()) for i in range(num_regions+1)])       
        self.classifiers =  nn.ModuleList([ nn.Linear(256+256, num_classes, bias = False) for i in range(num_regions+1)])
        self.s = 30.0
        
    def forward(self, x1, x2):

        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        
        
        B, C, H, W = x1.shape
        region_size = H // (self.num_regions // 2)

        patches1 = x1.unfold(2, region_size, region_size).unfold(3, region_size, region_size)
        patches1 = patches1.contiguous().view(B, C, -1, region_size, region_size)
        patches1 = patches1.permute(0, 2, 1, 3, 4)

        patches2 = x2.unfold(2, region_size, region_size).unfold(3, region_size, region_size)
        patches2 = patches2.contiguous().view(B, C, -1, region_size, region_size)
        patches2 = patches2.permute(0, 2, 1, 3, 4)

        output = []

        for i in range(self.num_regions):
            f1 = patches1[:,i,:,:,:] 
            f1 = self.eca[i](f1) 
            f1 = self.pool[i](f1).squeeze(3).squeeze(2) # vectorize it
            f1 =  self.region_net[i](f1)
            
            f2 = patches2[:,i,:,:,:] 
            f2 = self.eca[i](f2) 
            f2 = self.pool[i](f2).squeeze(3).squeeze(2) # vectorize it
            f2 =  self.region_net[i](f2)
            
            f = torch.cat((f1,f2),dim=1)

            f = self.s * self.classifiers[i](f)   
            output.append(f)

        output_stacked = torch.stack(output, dim = 2)

        # global part
        y1 = self.pool[4](self.eca[4](x1)).squeeze(3).squeeze(2)
        #y1 = self.globalavgpool[4](x1).squeeze(3).squeeze(2)     
        y1 = self.region_net[4](y1)
        
        y2 = self.pool[4](self.eca[4](x2)).squeeze(3).squeeze(2)
        #y2 = self.globalavgpool[4](x2).squeeze(3).squeeze(2)      
        y2 = self.region_net[4](y2)
          
        y = torch.cat((y1,y2),dim=1)                         
       
        output_global = self.s * self.classifiers[4](y).unsqueeze(2)
        output_final = torch.cat((output_stacked,output_global),dim=2)

        output_final = output_final.mean(dim=2)  # [B, num_classes] #teste

        return output_final
        
        




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)    







# I have to put the two inputs for training!!

class DualViewDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        x1, y1 = self.dataset1[idx]
        x2, y2 = self.dataset2[idx]
        assert y1 == y2
        return x1, x2, y1
    
def compute_class_weights(dataset, num_classes):
    counter = Counter()
    for _, label in dataset:
        counter[label] += 1

    total = sum(counter.values())
    weights = torch.zeros(num_classes)

    for c in range(num_classes):
        weights[c] = total / (counter[c] + 1e-6)

    return weights / weights.sum() * num_classes

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()




def main():
    seed = 26
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

        # Helper to count samples
    emotion_counts_train = {c: 0 for c in CLASSES}

    ROOT = Path.home() / "version_3/latest_3_0_ready_to_use_datasets"
    print("=== SCRIPT STARTED ===", flush=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        #transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    transform_augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
        degrees=5,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
        ),
        #transforms.Grayscale(),
        transforms.ToTensor(),

    ])

    train_dataset_1 = ConcatDataset([
        datasets.ImageFolder(ROOT / "AffectNet" / "train", transform=transform),
        #datasets.ImageFolder(ROOT / "CKplusIm" / "train", transform=transform),
        datasets.ImageFolder(ROOT / "Emo85" / "train", transform=transform),
        # datasets.ImageFolder(ROOT / "ExpWFormated" / "train", transform=transform), TRASH
        datasets.ImageFolder(ROOT / "FERPlus" / "train", transform=transform),
        datasets.ImageFolder(ROOT / "jaffeFormated" / "train", transform=transform),
        # datasets.ImageFolder(ROOT / "KDEFFormated" / "train", transform=transform),
        datasets.ImageFolder(ROOT / "MMAFEDB" / "train", transform=transform),
        datasets.ImageFolder(ROOT / "NHFI" / "train", transform=transform),
        datasets.ImageFolder(ROOT / "NONAMEFormated" / "train", transform=transform),
        datasets.ImageFolder(ROOT / "RAF-DB" / "train", transform=transform),
        datasets.ImageFolder(ROOT / "WSEFEPFormated" / "train", transform=transform)
    ])

    train_dataset_2 = ConcatDataset([
        datasets.ImageFolder(ROOT / "AffectNet" / "train", transform=transform_augmentation),
        #datasets.ImageFolder(ROOT / "CKplusIm" / "train", transform=transform_augmentation),
        datasets.ImageFolder(ROOT / "Emo85" / "train", transform=transform_augmentation),
        # datasets.ImageFolder(ROOT / "ExpWFormated" / "train", transform=transform_augmentation),
        datasets.ImageFolder(ROOT / "FERPlus" / "train", transform=transform_augmentation),
        datasets.ImageFolder(ROOT / "jaffeFormated" / "train", transform=transform_augmentation),
        # datasets.ImageFolder(ROOT / "KDEFFormated" / "train", transform=transform_augmentation),
        datasets.ImageFolder(ROOT / "MMAFEDB" / "train", transform=transform_augmentation),
        datasets.ImageFolder(ROOT / "NHFI" / "train", transform=transform_augmentation),
        datasets.ImageFolder(ROOT / "NONAMEFormated" / "train", transform=transform_augmentation),
        datasets.ImageFolder(ROOT / "RAF-DB" / "train", transform=transform_augmentation),
	datasets.ImageFolder(ROOT / "WSEFEPFormated" / "train" , transform= transform_augmentation)

    ])

    train_dual_dataset = DualViewDataset(train_dataset_1, train_dataset_2)
    train_loader = DataLoader(train_dual_dataset, batch_size=16, shuffle=True, num_workers=4,pin_memory=True)


    eval_dataset = [
        datasets.ImageFolder(ROOT / "AffectNet" / "eval", transform=transform),
        #datasets.ImageFolder(ROOT / "CKplusIm" / "eval", transform=transform),
        datasets.ImageFolder(ROOT / "Emo85" / "eval", transform=transform),
        # datasets.ImageFolder(ROOT / "ExpWFormated" / "train", transform=transform), TRASH
        datasets.ImageFolder(ROOT / "FERPlus" / "eval", transform=transform),
        datasets.ImageFolder(ROOT / "jaffeFormated" / "eval", transform=transform),
        # datasets.ImageFolder(ROOT / "KDEFFormated" / "train", transform=transform),
        datasets.ImageFolder(ROOT / "MMAFEDB" / "eval", transform=transform),
        datasets.ImageFolder(ROOT / "NHFI" / "eval", transform=transform),
        datasets.ImageFolder(ROOT / "NONAMEFormated" / "eval", transform=transform),
        datasets.ImageFolder(ROOT / "RAF-DB" / "eval", transform=transform),
        datasets.ImageFolder(ROOT / "WSEFEPFormated" / "eval", transform=transform) 
    ]
    eval_dataset  = ConcatDataset(eval_dataset)

    eval_dataloader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=4,pin_memory=True)

    print("Train samples:", len(train_dual_dataset))
    print("Eval samples:", len(eval_dataset))

    per_cls_weights = compute_class_weights(train_dataset_1, 6).to(device)

    model = CERN(num_classes=6, num_regions=4).to(device)

    criterion = FocalLoss(alpha=per_cls_weights, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    num_epochs = 100
    best_eval_acc = 0.0

    
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()
        train_correct = 0
        running_loss = 0.0
        train_total = 0

        for x1, x2, labels in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(x1, x2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            _, preds = outputs.max(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = 100 * train_correct / train_total
        running_loss /= train_total

        model.eval()
        eval_correct, total = 0, 0
        eval_loss = 0

        with torch.no_grad():
                for images, labels in eval_dataloader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images, images)
                    loss = criterion(outputs,labels)
                    eval_loss += loss.item() * labels.size(0)

                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    eval_correct += (predicted == labels).sum().item()
        eval_loss /= total
        eval_acc = 100.0 * eval_correct / total

        if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc

                

                torch.save(
                    model.state_dict(),
                    save_dir / "best_modelCERNA3Secondtry.pt"
                )


        print(f"Epoch {epoch+1} / train_Loss: {running_loss:.4f} / train_acc: {train_acc:.2f}% / eval_loss: {eval_loss:.2f} / eval_acc: {eval_acc:.2f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("structure2tryF")


if __name__ == "__main__":
    main()
