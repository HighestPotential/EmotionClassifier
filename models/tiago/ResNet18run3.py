
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset


import numpy as np
from matplotlib import pyplot as plt
from torch.optim import Adam

from tqdm import tqdm 
from torchvision import models
# from torchmetrics.classification import Accuracy
# from torchmetrics.classification import MulticlassAccuracy


from pathlib import Path

def main():
    ROOT = Path.home() / "ready_to_use_datasets"
    print("=== SCRIPT STARTED ===", flush=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    

  
    # acc = Accuracy(task="multiclass", num_classes=6).to(device) 


    num_classes = 6

    #Seting the model for ResNet18

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    

    # Adaptation for 64×64 images. From 224x224 to 64x64 in the first convolution, stride 1 (before 2) and 3x3 kernel instead of 7x7
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )

    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")


    model.maxpool = nn.Identity() # to prevent early downsample and feature loss

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)




        
    transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
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



    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    for d in train_dataset.datasets:
        print(d.class_to_idx)




    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()





    num_epochs = 50

    for epoch in range(num_epochs):
            model.train()
            train_correct = 0
            running_loss = 0
            train_total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)
                
                _, preds = outputs.max(1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            
            train_acc = 100 * train_correct / train_total
            running_loss /= train_total

            torch.save(
                    model.state_dict(),
                    save_dir / f"model_epoch_{epoch+1}.pt"
            )


            
            model.eval()
            eval_correct, total = 0, 0
            eval_loss = 0

            with torch.no_grad():
                for images, labels in test_dataloader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs,labels)
                    eval_loss += loss.item()

                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    eval_correct += (predicted == labels).sum().item()
            eval_loss /= total
            eval_acc = 100.0 * eval_correct / total

                   

            
            print(f"Epoch {epoch+1} / train_Loss: {running_loss:.4f} / train_acc: {train_acc:.2f}% / eval_loss: {eval_loss:.2f} / eval_acc: {eval_acc:.2f}")
            
   
           


    # to check accurac pytorch good or not
    # model.eval()
    # acc.reset()

    # with torch.no_grad():
    #     for images, labels in test_dataloader:
    #         images, labels = images.to(device), labels.to(device)

    #         outputs = model(images)
    #         preds = outputs.argmax(1)

    #         acc.update(preds, labels)

    # print("Test Accuracy:", acc.compute().item())

    # acc_per_class = MulticlassAccuracy(
    #     num_classes=6,
    #     average=None
    # ).to(device)



if __name__ == "__main__":
    main()

