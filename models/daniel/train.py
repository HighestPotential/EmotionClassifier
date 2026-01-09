# IMPORTS
import os
import time
from enum import Enum
from dataclasses import dataclass
from tqdm import tqdm

from FERDataset import FERDataset
import CustomModels as Models

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

#CONSTANTS
DATASETS_BASE: str = os.path.join("..", "..", "ready_to_use_datasets")
MAX_EPOCHS = 500

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# CLASSES & DATASTRUCTURES
class DataMode(Enum):
    train: str = "train"
    eval:  str = "eval"
    test:  str = "test"

@dataclass
class CNNContext:
    model: nn.Module
    criterion: nn.CrossEntropyLoss
    optimizer: optim.Optimizer
    device: torch.device
    threshold: float
    epochs: int
    patience: int
    saveFile: str

class NumberedList:
    def __init__(self, length):
        self.length = length
        self.data = [0 for _ in range(self.length)]

    def append(self, value):
        if len(self.data) >= self.length:
            self.data = self.data[1:]
        
        self.data.append(value)

    def min(self):
        if len(self.data) <= 0:
            return 0

        return min(self.data)
    
    def max(self):
        return max(self.data)
    
    def difference(self):
        return self.max() - self.min()
    
    def get(self):
        return self.data[-1]
    
# FUNCTIONS
def trainLoop(ctx: CNNContext,
              train_loader: DataLoader,
              val_loader: DataLoader,
              sampleList :NumberedList
              ) -> None:
    
    counter = 0
    best_loss = float('inf') # float rappresentation of infinity
    
    for epoch in range(ctx.epochs):

        print(f"TRAINING EPOCH {epoch}\n")

        for image, label in tqdm(train_loader, desc="Batch", leave=False):
            image = image.to(ctx.device)
            label = label.to(ctx.device)
            
            ctx.model.train()
            pred = ctx.model(image)

            loss = ctx.criterion(pred, label)
            
            loss = loss.to(ctx.device)

            ctx.optimizer.zero_grad()
            loss.backward()
            ctx.optimizer.step()


        ctx.model.eval()
        with torch.no_grad():
            runningLoss = 0
            correct = 0

            for X, y in val_loader:
                X = X.to(ctx.device)
                y = y.to(ctx.device)

                pred = ctx.model(X)
                loss = ctx.criterion(pred, y)
                runningLoss += float(loss.item())

                correct += (pred.argmax(1) == y).sum().item()

            epoch_loss = runningLoss / len(val_loader)
            epoch_acc = 100 * correct / len(val_loader.dataset)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ctx.model.state_dict(), ctx.saveFile)
        
        if sampleList.get() < ctx.threshold:
            counter += 1
        
        sampleList.append(epoch_loss)

        if ctx.patience < counter:
            print("Stopping training prematurely due to validation convergence!")
            break   # exit prematurely

        print(f"Validation Loss: {epoch_loss}")
        print(f"Validation Accuracy: {epoch_acc}\n")

def testAccuracy(ctx: CNNContext, test_loader: DataLoader) -> float:
    ctx.model.load_state_dict(torch.load(ctx.saveFile, weights_only=False, map_location=ctx.device))

    ctx.model.eval()
    with torch.no_grad():
        correct = 0
        for image, label in tqdm(test_loader, desc="Test progress", leave=False):
            if ctx.gpu:
                image = image.cuda()
                label = label.cuda()
            
            out = model(image)
            correct += (out.argmax(1) == label).sum().item()
        
    accuracy = 100 * correct / len(test_loader.dataset)
    
    return accuracy

if __name__ == "__main__":

    # INSTANCES
    trainSet = FERDataset(DATASETS_BASE, DataMode.train, transform)
    evalSet = FERDataset(DATASETS_BASE, DataMode.eval, transform)
    testSet = FERDataset(DATASETS_BASE, DataMode.test, transform)

    trainLoader = DataLoader(trainSet, batch_size=32, shuffle=True)
    evalLoader = DataLoader(evalSet, batch_size=32, shuffle=False)
    testLoader = DataLoader(testSet, batch_size=32, shuffle=False)

    #model = Models.BuildGoogLeNet(numClasses=6)    ACCURACY: 80%
    #model = Models.EmoNeXt_Tiny()                  ACCURACY: 71%
    
    Versions = {
        "Tiny": [
            [96, 192, 384, 768],
            [3, 3, 9, 3]
        ],
        "Small": [
            [96, 192, 384, 768],
            [3, 3, 27, 3]
        ],
        "Base": [
            [128, 256, 512, 1024],
            [3, 3, 27, 3]
        ],
        "Large": [
            [192, 384, 768, 1536],
            [3, 3, 27, 3]
        ],
        "XLarge": [
            [256, 512, 1024, 2048],
            [3, 3, 27, 3]
        ],
    }

    VersionToTrain = input("Enter Model Architecture (Tiny, Small, Base, Large, XLarge): ")
    if not VersionToTrain in Versions.keys():
        raise ValueError("Invalid input")

    Arch = Versions[VersionToTrain]

    model = Models.EmoNeXt_Variable(channels=Arch[0], blocks=Arch[1])
    loss_fn = nn.CrossEntropyLoss()
    optim_SGD = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optim_Adam = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #LOGIC
    model.to(device)

    trainCtx: CNNContext = CNNContext(model=model,
                                      criterion=loss_fn,
                                      optimizer=optim_SGD,
                                      device=device,
                                      threshold=1,        # Unit: %
                                      epochs=MAX_EPOCHS,
                                      patience=10,
                                      saveFile=f"./EmoNeXt_{VersionToTrain}_trained.pth"
                                    )
    
    accuraciesOverTime: NumberedList = NumberedList(10)

    start = time.time()
    trainLoop(ctx=trainCtx, train_loader=trainLoader, val_loader=evalLoader, sampleList=accuraciesOverTime)
    stop = time.time()
    duration = (stop - start) / 60 # duration in Minutes
    
    modelAccuracy = testAccuracy(ctx=trainCtx, test_loader=testLoader) 
    
    print(f"Model Accuracy: {modelAccuracy:.3f}%")
    print(f"Training took {duration:.2f} Minutes")