import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

batch_size = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = datasets.CIFAR10(root='./data/',
                               train=True,import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.CIFAR10(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6 ,3,1)
        self.conv2 = nn.Conv2d(6, 16, 3,1)
        self.conv3 = nn.Conv2d(16, 20, 3)
        self.conv4 = nn.Conv2d(20, 26, 3)
        self.conv5 = nn.Conv2d(26, 30, 3)
        self.conv6 = nn.Conv2d(30, 36, 3)
        self.conv7 = nn.Conv2d(36, 40, 3)
        self.conv8 = nn.Conv2d(40, 42, 3)
        self.conv9 = nn.Conv2d(42, 50, 3)
        self.conv10 = nn.Conv2d(50, 56,3)
        
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(20)
        self.bn4 = nn.BatchNorm2d(26)
        self.bn5 = nn.BatchNorm2d(30)
        self.bn6 = nn.BatchNorm2d(36)
        self.bn7 = nn.BatchNorm2d(40)
        self.bn8 = nn.BatchNorm2d(42)
        self.bn9 = nn.BatchNorm2d(50)
        self.bn10 = nn.BatchNorm2d(56)
        
        self.fc1 = nn.Linear(8064, 200)
        self.fc2 = nn.Linear(200, 10)
        

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = x.view(in_size,-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.to(device)), Variable(target.to(device))
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.data))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        with torch.no_grad():
            data, target = Variable(data.to(device)), Variable(target.to(device))
            output = model(data)
            test_loss += criterion(output,target).data
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, 101):
    train(epoch)
    test()

