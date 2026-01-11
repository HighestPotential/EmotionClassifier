
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset


import numpy as np
from matplotlib import pyplot as plt
from torch.optim import Adam


from pathlib import Path



class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()       
        self.feature_extractor = nn.Sequential(         
            nn.Conv2d(3, 32, kernel_size=3, padding=1), #(32,64,64)
            nn.ReLU(),
            
            nn.Conv2d(32,64, kernel_size=3, padding=1), #(64,64,64)
            nn.BatchNorm2d(64),             
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (64,32,32)

            nn.Conv2d(64,128, kernel_size=3, padding=1) ,
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Flatten() # [b, c,h,w] -> [b, (multiplication between c,h,w) ] (features might be lost. So it it being recommended to use AdaptiveAvgPool2d )
        )
        

        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )


    def forward(self, x):

        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x



def main():
    # seed = 26
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    ROOT = Path.home() / "ready_to_use_datasets"
    print("=== SCRIPT STARTED ===", flush=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_dataset = [
        datasets.ImageFolder(ROOT / "RAF-DB" / "train", transform=transform),
        datasets.ImageFolder(ROOT / "AffectNet" / "train", transform=transform),
        datasets.ImageFolder(ROOT / "FERPlus" / "train", transform=transform),
        datasets.ImageFolder(ROOT / "CKplusIm" / "train", transform=transform),
    ]

    train_dataset = ConcatDataset(train_dataset)
    


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4,pin_memory=True)

    test_dataset = [
        datasets.ImageFolder(ROOT / "RAF-DB" / "eval", transform=transform),
        datasets.ImageFolder(ROOT / "AffectNet" / "eval", transform=transform),
        datasets.ImageFolder(ROOT / "FERPlus" / "eval", transform=transform),
        datasets.ImageFolder(ROOT / "CKplusIm" / "eval", transform=transform),
    ]
    test_dataset  = ConcatDataset(test_dataset)

    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4,pin_memory=True)

    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))



    net = Net(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    for d in train_dataset.datasets:
        print(d.class_to_idx)



    

    for epoch in range(50):
        print(f"Epoch {epoch} started", flush=True)

        # TRAINING
        net.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        train_loss = 0
        

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            train_correct+= (preds == labels).sum().item()
            train_total += labels.size(0)
        
        running_loss /= train_total
        train_acc = 100.0 * train_correct / train_total
        train_loss = running_loss / train_total
        
        torch.save(
            net.state_dict(),
            save_dir / f"model_epoch_{epoch+1}.pt"
        )          


        # EVAL

        net.eval()
        eval_correct, eval_total = 0, 0
        eval_loss = 0.0

        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)

                eval_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                eval_total += labels.size(0)
                eval_correct += (predicted == labels).sum().item()

        eval_loss /= eval_total
        eval_acc = 100.0 * eval_correct / eval_total
               

        print(f"Epoch {epoch+1} / train_Loss: {train_loss:.4f} / train_acc: {train_acc:.2f}% / eval_loss: {eval_loss:.2f} / eval_acc: {eval_acc:.2f}")
            
     

if __name__ == "__main__":
    main()