
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import matplotlib as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim = 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())
x, y = Variable(x), Variable(y)

class Net(torch.nn.Module): #继承torch的Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() #继承_init_功能
        #定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(1, 10, 1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr = 0.5)
loss_function = torch.nn.MSELoss()
#实时打印
plt.ion()
plt.show()

for t in range(100):
    prediction = net(x)
    loss = loss_function(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 5 == 0:
    #plot and show learning process
    plt.cla()
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), y.data.numpy(), 'r-', lw = 5)
    plt.text(0.5, 0, 'Loss = %.4f'%loss.data[0], fontdict = {'size':20, 'color': 'red'})
    plt.pause(0.1)

    
#Classifier
n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)