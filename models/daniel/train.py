# IMPORTS
import os
from enum import Enum
from dataclasses import dataclass
from tqdm import tqdm

from FERDataset import FERDataset
from CustomModels import DeepEmotion

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

#CONSTANTS
DATASETS_BASE: str = os.path.join("..", "..", "ready_to_use_datasets")
MAX_EPOCHS = 100

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
    gpu: bool
    threshold: float
    epochs: int
    patience: int

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
              lossList :NumberedList
              ):
    
    counter = 0
    best_loss = float('inf')
    
    for epoch in range(ctx.epochs):

        print(f"TRAINING EPOCH {epoch}\n")

        for image, label in tqdm(train_loader, desc="Batch", leave=False):
            if ctx.gpu:
                image = image.cuda()
                label = label.cuda()
            
            ctx.model.train()
            pred = ctx.model(image)

            loss = ctx.criterion(pred, label)
            
            if ctx.gpu:
                loss = loss.cuda()

            ctx.optimizer.zero_grad()
            loss.backward()
            ctx.optimizer.step()


        ctx.model.eval()
        with torch.no_grad():
            runningLoss = 0
            correct = 0

            for X, y in val_loader:
                if ctx.gpu:
                    X = X.cuda()
                    y = y.cuda()

                pred = ctx.model(X)
                loss = ctx.criterion(pred, y)
                correct += (pred.argmax(1) == y).sum().item()

                runningLoss += loss.item()

            epoch_loss = runningLoss / len(val_loader.dataset)
            epoch_acc = 100 * correct / len(val_loader.dataset)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ctx.model.state_dict(), "DeepEmotion_trained.pth")
        
        if lossList.get() + ctx.threshold < epoch_loss:
            counter += 1
        
        lossList.append(epoch_loss)

        if counter > ctx.patience:
            break   # exit prematurely

        print(f"Validation Loss: {epoch_loss}")
        print(f"Validation Accuracy: {epoch_acc}\n")

    return counter

def testAccuracy(ctx: CNNContext, test_loader: DataLoader, parameterPath: str):
    ctx.model.load_state_dict(torch.load(parameterPath, weights_only=False))

    ctx.model.eval()
    with torch.no_grad():
        correct = 0
        for image, label in test_loader:
            if ctx.gpu:
                image = image.cuda()
                label = label.cuda()
            
            out = model(image)
            correct += (out.argmax(1) == label).sum().item()
        
    accuracy = correct / len(test_loader)
    
    return accuracy

if __name__ == "__main__":

    # INSTANCES
    trainSet = FERDataset(DATASETS_BASE, DataMode.train, transform)
    evalSet = FERDataset(DATASETS_BASE, DataMode.eval, transform)
    testSet = FERDataset(DATASETS_BASE, DataMode.test, transform)

    trainLoader = DataLoader(trainSet, batch_size=32, shuffle=True)
    evalLoader = DataLoader(evalSet, batch_size=32, shuffle=False)
    testLoader = DataLoader(testSet, batch_size=32, shuffle=False)

    model = DeepEmotion()
    loss_fn = nn.CrossEntropyLoss()
    optim_SGD = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optim_Adam = optim.Adam(model.parameters(), lr=0.001)

    use_gpu = torch.cuda.is_available()

    #LOGIC
    if use_gpu:
        model.cuda()

    trainCtx: CNNContext = CNNContext(model=model,
                                      criterion=loss_fn,
                                      optimizer=optim_Adam,
                                      gpu=use_gpu,
                                      threshold=0.001,
                                      epochs=MAX_EPOCHS,
                                      patience=10)
    
    accuraciesOverTime: NumberedList = NumberedList(5)
    over = trainLoop(ctx=trainCtx, train_loader=trainLoader, val_loader=evalLoader, lossList=accuraciesOverTime)
    acc = testAccuracy(ctx=trainCtx, test_loader=testLoader, parameterPath="./DeepEmotion_trained.pth")
    print(f"Model Accuracy: {acc:.3f}%")