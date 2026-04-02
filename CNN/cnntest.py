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

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

########################################################################
# .. note::
#     If you are running this tutorial on Windows or MacOS and encounter a
#     BrokenPipeError or RuntimeError related to multiprocessing, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

#定义预处理的流程，先将图片转换成Tensor，再进行归一化处理。
transform = transforms.Compose(
    [transforms.ToTensor(), #把图片转成 PyTorch 的张量（一种深度学习可以使用的数据集）
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 把数据归一化（-1，1）均值是 (0.5, 0.5, 0.5)，标准差（每一个数据和均值的差的平方的平均值的平方根）是 (0.5, 0.5, 0.5)

batch_size = 4 #每次训练的图片数量

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

########################################################################
# Let us show some of the training images, for fun.

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

# 显示数据用的
def imshow(img):
    img = img / 2 + 0.5     # unnormalize 把数据还原成（0，1）之间的数
    npimg = img.numpy()     # 把张量转换成数组
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # matplotlib 的图片需要[高，宽，通道] ，原始数据是【通道，高，宽】转换一下
    plt.show()   #matplotlib 显示图片


# get some random training images
dataiter = iter(trainloader) # 把训练集转换成一个迭代器，方便后期去取数据，去数据间隔 batch_size
images, labels = next(dataiter) # 取出一个 batch_size 的数据

print('Images show start')
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
print('Images show end')


########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn # 导入神经网络模块，包含各种神经网络层（卷积，线性，全连接）损失函数。模块基类
import torch.nn.functional as F # 导入了函数各种函数的包，比如激活函数，损失函数，卷积操作等


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


net = Net() #创建了一个实例，net 就是一个神经网络对象

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim # 导入优化器模块，比如梯度下降优化算法

criterion = nn.CrossEntropyLoss() # 定义损失函数，交叉熵损失
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

for epoch in range(2):  # loop over the dataset multiple times // 训练两轮

    running_loss = 0.0 #统计完整一轮的损失
    for i, data in enumerate(trainloader, 0): #从 0 开始枚举 trainloader 中的数据，每组 4 张图片
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data # inputs 是数据图片，labels 是图片对应的标签

        # zero the parameter gradients
        optimizer.zero_grad() # 梯度清零，每一批数据都是走一步，计算一下梯度，防止梯度累计，造成和预期结果不符合的情况。

        # forward + backward + optimize
        outputs = net(inputs) #向前传播，算出当前的输出（初始数据是乱码）
        loss = criterion(outputs, labels) #计算出损失，即输出和标准标签的差距
        loss.backward() # 反向传播，计算梯度
        optimizer.step() # 根据上面的梯度，优化一波内置的参数

        # print statistics
        running_loss += loss.item() # 累计损失
        if i % 2000 == 1999:    # print every 2000 mini-batches 每 2000 批数据打印一次损失平均值
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training') #训练到此结束

########################################################################
# Let's quickly save our trained model:

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH) #保存模型，net.state_dict() 是一个字典对象，包含了模型的所有参数（权重和偏置），PATH 是保存模型的路径，可以自己定义

########################################################################
# See `here <https://pytorch.org/docs/stable/notes/serialization.html>`_
# for more details on saving PyTorch models.
#
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

# 开始正式测试，查看训练结果
dataiter = iter(testloader) #把测试集转换成一个迭代器，方便后期去取数据，去数据间隔 batch_size
images, labels = next(dataiter) # 取出一个 batch_size 的数据，images 是数据图片，labels 是图片对应的标签

# print images
imshow(torchvision.utils.make_grid(images)) #显示一个多张图片拼成的网格图（make_grid 把图片拼接成一张图）
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4))) # 打印对应的标签 
#' '.join() 是把列表中的元素用指定的字符连接起来, 
# f'{classes[labels[j]]:5s}' 是把标签转换成字符串，并格式化成 5 个字符，j 是图片的索引,
# for j in range(4) 是循环 4 张图片，打印对应的标签

########################################################################
# Next, let's load back in our saved model (note: saving and re-loading the model
# wasn't necessary here, we only did it to illustrate how to do so):

net = Net()  #创建一个神经网络实例
net.load_state_dict(torch.load(PATH, weights_only=True)) #加载模型，不加载优化器状态，训练步数等信息。如果为 false 一般是为了恢复训练，如果是单纯的模型参数使用，那么可以设置为 true

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

outputs = net(images) #向前传播，算出对应的数据的输出

########################################################################
# The outputs are energies for the 10 classes.
# The higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs, 1) #获取最大值的索引

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

correct = 0 # 统计正确的数量
total = 0 # 统计总数量
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad(): #关闭梯度计算的情况下，处理一下数据
    for data in testloader: #遍历测试集
        images, labels = data #获取图片和标签
        # calculate outputs by running images through the network
        outputs = net(images) #向前传播，计算推导结果
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1) #获取最大值索引
        total += labels.size(0) #统计总数量
        correct += (predicted == labels).sum().item() #统计正确的数量

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

########################################################################
# That looks way better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes} #初始化一个关于类别的字典，默认值为 0
total_pred = {classname: 0 for classname in classes} #初始化关于类别的字典，默认值为 0，计算总的各类别总数

# again no gradients needed
with torch.no_grad(): #关闭梯度计算，做以下操作
    for data in testloader: #遍历测试集
        images, labels = data #获取图片数据
        outputs = net(images) #向前传播，计算推导结果
        _, predictions = torch.max(outputs, 1) #获取最大值
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions): #同时遍历标签和预测结果，统计每个类别的正确预测数量和总数量
            # zip 把 labels 和 predictions 两个列表对应的位置的数据提取出来配对，zip 后的列表长度为两个列表长度的较小值
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items(): #遍历每个类别和对应的正确预测数量，计算每个类别的准确率
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

########################################################################
# Okay, so what next?
#
# How do we run these neural networks on the GPU?
#
# Training on GPU
# ----------------
# Just like how you transfer a Tensor onto the GPU, you transfer the neural
# net onto the GPU.
#
# Let's first define our device as the first visible cuda device if we have
# CUDA available:

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

########################################################################
# The rest of this section assumes that ``device`` is a CUDA device.
#
# Then these methods will recursively go over all modules and convert their
# parameters and buffers to CUDA tensors:
#
# .. code:: python
#
#     net.to(device)
#
#
# Remember that you will have to send the inputs and targets at every step
# to the GPU too:
#
# .. code:: python
#
#         inputs, labels = data[0].to(device), data[1].to(device)
#
# Why don't I notice MASSIVE speedup compared to CPU? Because your network
# is really small.
#
# **Exercise:** Try increasing the width of your network (argument 2 of
# the first ``nn.Conv2d``, and argument 1 of the second ``nn.Conv2d`` –
# they need to be the same number), see what kind of speedup you get.
#
# **Goals achieved**:
#
# - Understanding PyTorch's Tensor library and neural networks at a high level.
# - Train a small neural network to classify images
#
# Training on multiple GPUs
# -------------------------
# If you want to see even more MASSIVE speedup using all of your GPUs,
# please check out :doc:`data_parallel_tutorial`.
#
# Where do I go next?
# -------------------
#
# -  :doc:`Train neural nets to play video games </intermediate/reinforcement_q_learning>`
# -  `Train a state-of-the-art ResNet network on imagenet`_
# -  `Train a face generator using Generative Adversarial Networks`_
# -  `Train a word-level language model using Recurrent LSTM networks`_
# -  `More examples`_
# -  `More tutorials`_
# -  `Discuss PyTorch on the Forums`_
# -  `Chat with other users on Slack`_
#
# .. _Train a state-of-the-art ResNet network on imagenet: https://github.com/pytorch/examples/tree/master/imagenet
# .. _Train a face generator using Generative Adversarial Networks: https://github.com/pytorch/examples/tree/master/dcgan
# .. _Train a word-level language model using Recurrent LSTM networks: https://github.com/pytorch/examples/tree/master/word_language_model
# .. _More examples: https://github.com/pytorch/examples
# .. _More tutorials: https://github.com/pytorch/tutorials
# .. _Discuss PyTorch on the Forums: https://discuss.pytorch.org/
# .. _Chat with other users on Slack: https://pytorch.slack.com/messages/beginner/

# %%%%%%INVISIBLE_CODE_BLOCK%%%%%%
del dataiter
# %%%%%%INVISIBLE_CODE_BLOCK%%%%%%