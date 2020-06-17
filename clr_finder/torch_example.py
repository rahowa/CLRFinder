import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torchvision.models import squeezenet
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage
from torch.utils.data import DataLoader
from torch_pipeline import LRFinderPytorch, PytorchScheduler
from scalers import Scaler


class TestModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.ga_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear1 = nn.Linear(64, 10)
        self.softmax = nn.LogSoftmax(1)

    def forward(self, x):
        x = F.relu_(self.bn1(F.max_pool2d(self.conv1(x), 2)))
        x = F.relu_(self.bn2(F.max_pool2d(self.conv2(x), 2)))
        x = F.relu_(self.bn3(F.max_pool2d(self.conv3(x), 2)))

        x = self.ga_pooling(x)
        x = x.view(-1, 64)
        x = self.linear1(x)
        x = self.softmax(x)
        return x


def train(model, criterion, optimizer, loader, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        epoch_loss += loss.item()
        epoch_acc += output.max(1)[1].eq(target).sum().item()
        loss.backward()
        optimizer.step()
    return model, epoch_acc / len(loader.dataset), epoch_loss / len(loader)


if __name__ == '__main__':
    model = TestModel()
    model.to('cpu')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), 1.0)
    
    transforms = Compose([ToTensor()])
    train_dataset = MNIST('./', train=True, download=True, transform=transforms)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

    finder = LRFinderPytorch(model=model, min_lr=1e-3, max_lr=0.9, loader=train_loader, n_epochs=1)
    finder.run(criterion, optimizer)
    # finder.plot_result()
    best_lr = finder.best_lr()
    scheduler = PytorchScheduler(best_lr[0], best_lr[1], Scaler.triangle1, len(train_loader), 1.)
    
    optimizer = optim.Adam(model.parameters(), best_lr[1])
    for ep in range(10):
        model, acc, loss = train(model, criterion, optimizer, train_loader, 'cpu')
        print(f"Epooch: {ep}, loss: {loss}, acc: {acc}")


    