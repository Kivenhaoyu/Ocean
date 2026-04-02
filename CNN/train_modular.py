# -*- coding: utf-8 -*-
"""
Training a Classifier
=====================

This is it. You have seen how to define neural networks, compute loss and make
updates to the weights of the network.

Now you might be thinking,

What about data?
----------------

Generally, when you have to deal with image, text, audio or video data,
you can use standard python packages that load data into a numpy array.
Then you can convert this array into a ``torch.*Tensor``.

-  For images, packages such as Pillow, OpenCV are useful
-  For audio, packages such as scipy and librosa
-  For text, either raw Python or Cython based loading, or NLTK and
   SpaCy are useful

Specifically for vision, we have created a package called
``torchvision``, that has data loaders for common datasets such as
ImageNet, CIFAR10, MNIST, etc. and data transformers for images, viz.,
``torchvision.datasets`` and ``torch.utils.data.DataLoader``.

This provides a huge convenience and avoids writing boilerplate code.

For this tutorial, we will use the CIFAR10 dataset.
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,
‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

.. figure:: /_static/img/cifar10.png
   :alt: cifar10

   cifar10


Training an image classifier
----------------------------

We will do the following steps in order:

1. Load and normalize the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

1. Load and normalize CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using ``torchvision``, it’s extremely easy to load CIFAR10.
"""
import torch  #深度学习的框架，提供张量计算
import torchvision #专门用来预处理图片数据集和预处理的工具包
import torchvision.transforms as transforms #图片预处理工具包
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn # 导入神经网络模块，包含各种神经网络层（卷积，线性，全连接）损失函数。模块基类
import torch.nn.functional as F # 导入了函数各种函数的包，比如激活函数，损失函数，卷积操作等
import torch.optim as optim # 导入优化器模块，比如梯度下降优化算法

# batch_size:每次训练的图片数量
def build_dataloaders(batch_size=4) :
#定义预处理的流程，先将图片转换成Tensor，再进行归一化处理。
    transform = transforms.Compose(
        [transforms.ToTensor(), #把图片转成 PyTorch 的张量（一种深度学习可以使用的数据集）
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 把数据归一化（-1，1）均值是 (0.5, 0.5, 0.5)，标准差（每一个数据和均值的差的平方的平均值的平方根）是 (0.5, 0.5, 0.5)

    # 加载训练集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
    # 加载测试集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes

class Net(nn.Module): # 定义神经网络，nn.Module 是所有神经网络的基类
    def __init__(self): #构造函数，初始化
        super().__init__() # 调用父类初始化函数
        self.conv1 = nn.Conv2d(3, 6, 5) # 第一层卷积层，输入通道数是3，输出通道数是6，卷积核大小是5，步长是1，填充是0 
        #原始数据 3*32*32，经过第一层卷积后数据为6*(32-5+1)*(32-5+1)
        self.pool = nn.MaxPool2d(2, 2) # 最大池化层，池化核大小是2，步长是2
        #第一层卷积后的数据 6*(32-5+1)*(32-5+1)经过池化后数据为 6*(32-5+1)/2*(32-5+1)/2
        self.conv2 = nn.Conv2d(6, 16, 5)# 第二层卷积层，输入通道数是6，输出通道数是16，卷积核大小是5，步长是1，填充是0
        #经过第二层卷积后数据为 16*((32-5+1)/2-5+1)*((32-5+1)/2-5+1) = 16*10*10
        #在 forward 145行代码，第二层卷积也通过了 pool 的池化，所以是 16*10/2*10/2 = 16*5*5
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 全连接层，输入通道数是16*5*5，输出通道数是120，120是输出通道数，也就是全连接层的神经元数量（隐藏神经元，即不是输入也不是输出的神经元）
        self.fc2 = nn.Linear(120, 84) # 全连接层，输入通道数是120，输出通道数是84，84是输出通道数，也就是全连接层的神经元数量（隐藏神经元，即不是输入也不是输出的神经元）
        self.fc3 = nn.Linear(84, 10)    # 全连接层，输入通道数是84，输出通道数是10，10是最后的分类，所以最后输出的通道数一定为具体的分类数量

# CNN 向前传播过程函数实现
    def forward(self, x): # 前向传播函数，重点介绍，Torch 神经网络特殊函数，定义了神经网络的前向传播过程，输入是 x（张量），输出是 x。不需要用 net.forward(x) 调用，直接 net(x) 就可以了，PyTorch 会自动调用 forward 函数
        x = self.pool(F.relu(self.conv1(x))) # 第一层卷积->激活->池化
        x = self.pool(F.relu(self.conv2(x))) # 第二层卷积->激活->池化
        x = torch.flatten(x, 1) # flatten all dimensions except batch   # 把卷积层的输出展平，变成全连接层的输入，batch_size*16*5*5 变成 batch_size*400
        x = F.relu(self.fc1(x)) # 第一层全连接->激活（激活是为了增加表达能力适合更复杂的函数）
        x = F.relu(self.fc2(x)) # 第二层全连接->激活（激活是为了增加表达能力适合更复杂的函数）
        x = self.fc3(x) # 最后一层全连接
        return x

def build_model(config):
    if config['model'] == 'cnn':
        return Net() #创建了一个实例，net 就是一个神经网络对象
    elif config['model'] == 'rnn':
        raise NotImplementedError("rnn not implemented yet")
    elif config['model'] == 'transformer':
        raise NotImplementedError("transformer not implemented yet")

def train_one_epoch(model,loader, optimizer, criterion, device):
    model.train() #把模型设置成训练模式
    running_loss=0.0 #统计完整一轮的损失
    total_samples = 0
    for data in loader: #从 0 开始枚举 trainloader 中的数据，每组 4 张图片
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data # inputs 是数据图片，labels 是图片对应的标签
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad() # 梯度清零，每一批数据都是走一步，计算一下梯度，防止梯度累计，造成和预期结果不符合的情况。

        # forward + backward + optimize
        outputs = model(inputs) #向前传播，算出当前的输出（初始数据是乱码）
        loss = criterion(outputs, labels) #计算出损失，即输出和标准标签的差距
        loss.backward() # 反向传播，计算梯度
        optimizer.step() # 根据上面的梯度，优化一波内置的参数

        batch_size_now = labels.size(0) #当前批次的大小，最后一个批次可能小于 batch_size
        running_loss += loss.item() * batch_size_now # 累计损失，乘以当前批次的大小，得到当前批次的总损失
        total_samples += batch_size_now #累计样本数量
    avg_loss = running_loss / total_samples
    return avg_loss

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    correct = 0
    with torch.no_grad(): #关闭梯度计算的情况下，处理一下数据
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size_now = labels.size(0)
            running_loss += loss.item() * batch_size_now
            total_samples += batch_size_now
            _, predicted = torch.max(outputs, 1) #获取最大值索引
            correct += (predicted == labels).sum().item() #统计正确的数量

        avg_loss = running_loss / total_samples
        accuracy = correct / total_samples
    return avg_loss, accuracy
batch_size = 4
trainloader, testloader, classes = build_dataloaders(batch_size)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = build_model(config = {'model': 'cnn'})
net = net.to(device)
criterion = nn.CrossEntropyLoss() # 定义损失函数，交叉熵损失
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(2):  # loop over the dataset multiple times // 训练两轮
    train_loss = train_one_epoch(net, trainloader, optimizer, criterion, device) #训练一轮
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.3f}')

eval_loss, accuracy = evaluate(net, testloader, criterion, device)
print(f'Epoch {epoch + 1}: train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}, eval_acc={accuracy * 100:.2f}%')
