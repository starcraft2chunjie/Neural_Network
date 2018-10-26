"""Some of the models:
   VGG-16: conv = 3 * 3 filter, s = 1, same
           max-pool = 2 * 2, s = 2
           16 refers that this has 16 weight layer
   ResNet: 1. residual block
              a(l+2) = g(z(l + 2) + a (l))
              It's called 'short cut'/'skip connection' which
              allows you train deeper neural network
           2. then you need to stack residual network into your neural network
           advantage: It doesn't hurt to make deeper neural network
           reason: a(l+2) = g(W(l+2) * a(l+1) + a(l)), you can be easier to make
                   the W and b to be 0, then g(a(l)) = a(l), so it's easier to 
                   an identity function to guarantee the performance doesn't hurt
                   , and you cabe lucky to get better performance.
           note: maybe you need to add a W before the a(l) to fir the dimension
    inception network:
           for one layer, you need to try 1 * 1 concolution and 3 * 3 and 5 * 5
           to get the same dimension as the layer before, then you concat these
           these together. Because we want to reduce the computational cost, 
           for 3 * 3 convolution, you can insert a 1 * 1 convolution, then you 
           reduce the cost.
    Some of the method:
    image augmentation:
           mirroring and random crooping
           rotation and shearing(used less because of the complexity)
           color shifting
           PCA color augmentation:
           reduce the colour which is strong and strengthen the weak color,
           in order to make a balance situation."""

from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root = './data', train = True, download = False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#show some of the training image for fun
import matplotlib.pyplot as plt
import numpy as np 
def imshow(img):
    img = img/2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
#show images
imshow(torchvision.utils.make_grid(images))
#Print labels
print(''.join('%5s' % classes[labels[j]] for j in range(4)))
#define a Convolution Neural network
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
#define a loss function and optimizer
#Let's use a Classification Cross-Entropy loss and SGD with momentum
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
#train the network
#We simply have to loop over our data iterator, and feed the inputs
#to the network and optimize
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        #get the inputs
        inputs, labels = data
        #wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        #zero the parameter gradients
        optimizer.zero_grad()
        #forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step() #does the update
        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999: #print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/2000))
            running_loss = 0.0
print('finishing training')

