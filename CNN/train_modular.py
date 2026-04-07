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
from copy import deepcopy
from itertools import product

# batch_size:每次训练的图片数量
def build_dataloaders(batch_size=4, validation_split=0.1, seed=42, use_validation=True):
#定义预处理的流程，先将图片转换成Tensor，再进行归一化处理。
    transform = transforms.Compose(
        [transforms.ToTensor(), #把图片转成 PyTorch 的张量（一种深度学习可以使用的数据集）
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 把数据归一化（-1，1）均值是 (0.5, 0.5, 0.5)，标准差（每一个数据和均值的差的平方的平均值的平方根）是 (0.5, 0.5, 0.5)

    # 加载训练集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    # 加载测试集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    valloader = None
    if use_validation and 0 < validation_split < 1:
        total_samples = len(trainset)
        val_size = int(total_samples * validation_split)
        train_size = total_samples - val_size
        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = torch.utils.data.random_split(
            trainset,
            [train_size, val_size],
            generator=generator,
        )
        trainloader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        valloader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, valloader, testloader, classes

class CNNNet(nn.Module): # 定义神经网络，nn.Module 是所有神经网络的基类
    def __init__(self, config): #构造函数，初始化
        super().__init__() # 调用父类初始化函数
        conv1_out = config['conv1_out']
        conv2_out = config['conv2_out']
        fc1_hidden = config['fc1_hidden']
        fc2_hidden = config['fc2_hidden']

        self.conv1 = nn.Conv2d(3, conv1_out, 5) # 第一层卷积层，输入通道数是3，输出通道数可配置，卷积核大小是5，步长是1，填充是0 
        #原始数据 3*32*32，经过第一层卷积后数据为6*(32-5+1)*(32-5+1)
        self.pool = nn.MaxPool2d(2, 2) # 最大池化层，池化核大小是2，步长是2
        #第一层卷积后的数据 6*(32-5+1)*(32-5+1)经过池化后数据为 6*(32-5+1)/2*(32-5+1)/2
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, 5)# 第二层卷积层，输入通道数来自第一层，输出通道数可配置，卷积核大小是5，步长是1，填充是0
        #经过第二层卷积后数据为 16*((32-5+1)/2-5+1)*((32-5+1)/2-5+1) = 16*10*10
        #在 forward 145行代码，第二层卷积也通过了 pool 的池化，所以是 16*10/2*10/2 = 16*5*5
        self.fc1 = nn.Linear(conv2_out * 5 * 5, fc1_hidden) # 全连接层输入维度跟着卷积输出走，隐藏层大小可配置
        self.fc2 = nn.Linear(fc1_hidden, fc2_hidden) # 第二层全连接隐藏层大小可配置
        self.fc3 = nn.Linear(fc2_hidden, 10)    # 全连接层，输出通道数是10，10是最后的分类，所以最后输出的通道数一定为具体的分类数量

# CNN 向前传播过程函数实现
    def forward(self, x): # 前向传播函数，重点介绍，Torch 神经网络特殊函数，定义了神经网络的前向传播过程，输入是 x（张量），输出是 x。不需要用 net.forward(x) 调用，直接 net(x) 就可以了，PyTorch 会自动调用 forward 函数
        x = self.pool(F.relu(self.conv1(x))) # 第一层卷积->激活->池化
        x = self.pool(F.relu(self.conv2(x))) # 第二层卷积->激活->池化
        x = torch.flatten(x, 1) # flatten all dimensions except batch   # 把卷积层的输出展平，变成全连接层的输入，batch_size*16*5*5 变成 batch_size*400
        x = F.relu(self.fc1(x)) # 第一层全连接->激活（激活是为了增加表达能力适合更复杂的函数）
        x = F.relu(self.fc2(x)) # 第二层全连接->激活（激活是为了增加表达能力适合更复杂的函数）
        x = self.fc3(x) # 最后一层全连接
        return x

MODEL_REGISTRY = {
    'cnn': CNNNet,
}


def build_model(config):
    model_name = config['model']
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name](config)
    if model_name == 'rnn':
        raise NotImplementedError('rnn model is reserved but not implemented yet')
    if model_name == 'transformer':
        raise NotImplementedError('transformer model is reserved but not implemented yet')
    raise ValueError(f"unsupported model: {model_name}")


def build_optimizer(model, config):
    optimizer_name = config.get('optimizer', 'sgd').lower()
    lr = config.get('lr', 0.001)

    if optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=config.get('momentum', 0.9))
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    raise ValueError(f"unsupported optimizer: {optimizer_name}")


def build_criterion(config):
    loss_name = config.get('loss', 'cross_entropy').lower() #获取损失函数， lower() 转换为小写字母
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    raise ValueError(f"unsupported loss: {loss_name}")

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


#默认配置，后续会在搜索空间中覆盖这些默认配置
DEFAULT_CONFIG = {
    'name': 'baseline-cnn',
    'model': 'cnn',
    'conv1_out': 6,
    'conv2_out': 16,
    'fc1_hidden': 120,
    'fc2_hidden': 84,
    'batch_size': 4,
    'epochs': 2,
    'lr': 0.001,
    'momentum': 0.9,
    'optimizer': 'sgd',
    'loss': 'cross_entropy', # 指定损失函数
    'validation_split': 0.1, # 训练集划分出 10% 作为验证集
    'seed': 42, # 随机数种子，保证每次运行结果一致
    # 随机数生成器：伪随机数，从一个初始值按照一定规律生成随机数，固定初始值，后期使用的随机数就是一定的
}

# 搜索空间，定义了每个超参数的候选值，训练过程中会从这些候选值中进行组合，形成不同的配置进行训练和评估
SEARCH_SPACE = {
    'model': ['cnn'],
    'conv1_out': range(6, 11), # 第一层卷积核数量
    'conv2_out': [16], # 第二层卷积核数量
    'fc1_hidden': [120], #第一层全连接隐藏输出大小
    'fc2_hidden': [84], #第二层全连接隐藏输出大小
    'batch_size': [4, 16], # 批次数量
    'epochs': [2], # 训练轮数
    'optimizer': ['sgd'], # 优化器
    'lr': [0.001, 0.005], #学习率，每次
    'momentum': [0.9], #优化器辅助动量
    # 参数的更新 = momentum * velocity + lr * gradient (辅助动量*之前的更新 + 学习率*当前的梯度)
}


# 合并配置
def merge_config(base_config, overrides):
    merged = deepcopy(base_config)
    merged.update(overrides)
    return merged


def set_seed(seed):
    torch.manual_seed(seed) # PyTorch 的随机数生成器，生成随机数的算法，固定种子，保证每次生成的随机数一样
    if torch.cuda.is_available(): # 如果 GPU 可用，则设置所有 GPU 的种子
        torch.cuda.manual_seed_all(seed)


# 生成所有可能的配置
def generate_search_configs(base_config, search_space):
    keys = list(search_space.keys()) #获取搜索空间中的所有键
    values = [search_space[key] for key in keys] #获取搜索空间中的所有值
    configs = []

    for trial_id, combination in enumerate(product(*values), start=1):
        overrides = dict(zip(keys, combination)) # 将键和可能设置的一种可能性值组合成字典
        config = merge_config(base_config, overrides)
        config['name'] = f"trial-{trial_id:02d}"
        configs.append(config)

    return configs


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def run_experiment(config, use_validation=True, evaluate_test=False):
    set_seed(config['seed'])
    trainloader, valloader, testloader, _ = build_dataloaders(
        batch_size=config['batch_size'],
        validation_split=config['validation_split'],
        seed=config['seed'],
        use_validation=use_validation,
    )
    device = get_device()
    model = build_model(config).to(device)
    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)

    train_loss = None
    for epoch in range(config['epochs']):
        train_loss = train_one_epoch(model, trainloader, optimizer, criterion, device)
        print(
            f"[{config['name']}] epoch {epoch + 1}/{config['epochs']} "
            f"train_loss={train_loss:.4f}"
        )

    val_loss = None
    val_acc = None
    if use_validation and valloader is not None:
        val_loss, val_acc = evaluate(model, valloader, criterion, device)

    test_loss = None
    test_acc = None
    if evaluate_test:
        test_loss, test_acc = evaluate(model, testloader, criterion, device)

    result = {
        'name': config['name'],
        'model': config['model'],
        'conv1_out': config['conv1_out'],
        'conv2_out': config['conv2_out'],
        'fc1_hidden': config['fc1_hidden'],
        'fc2_hidden': config['fc2_hidden'],
        'batch_size': config['batch_size'],
        'epochs': config['epochs'],
        'lr': config['lr'],
        'optimizer': config['optimizer'],
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
    }
    return result


def print_comparison(results, metric_key='val_acc'):
    print('\nHyperparameter Search Results')
    print('-' * 152)
    print(
        f"{'name':<18}{'model':<12}{'c1':<6}{'c2':<6}{'fc1':<8}{'fc2':<8}{'batch':<8}{'lr':<10}"
        f"{'optimizer':<12}{'train_loss':<12}{'val_loss':<12}{'val_acc':<10}{'test_acc':<10}"
    )
    print('-' * 152)
    sorted_results = sorted(
        results,
        key=lambda item: item[metric_key] if item[metric_key] is not None else float('-inf'),
        reverse=True,
    )
    for result in sorted_results:
        val_loss = result['val_loss'] if result['val_loss'] is not None else float('nan')
        val_acc = result['val_acc'] * 100 if result['val_acc'] is not None else float('nan')
        test_acc = result['test_acc'] * 100 if result['test_acc'] is not None else float('nan')
        print(
            f"{result['name']:<18}{result['model']:<12}{result['conv1_out']:<6}{result['conv2_out']:<6}"
            f"{result['fc1_hidden']:<8}{result['fc2_hidden']:<8}{result['batch_size']:<8}"
            f"{result['lr']:<10.4f}{result['optimizer']:<12}{result['train_loss']:<12.4f}"
            f"{val_loss:<12.4f}{val_acc:<10.2f}{test_acc:<10.2f}"
        )


def find_best_result(results, metric_key='val_acc'):
    return max(
        results,
        key=lambda item: item[metric_key] if item[metric_key] is not None else float('-inf'),
    )


def train_best_config_on_full_data(config):
    final_config = merge_config(config, {'name': f"{config['name']}-final"})
    result = run_experiment(final_config, use_validation=False, evaluate_test=True)
    return result

import optuna
import time

def objective(trial):
    config = {
        'name': f"optuna-trial-{trial.number}",
        'model': 'cnn',
        'conv1_out': trial.suggest_int('conv1_out', 6, 12),
        'conv2_out': trial.suggest_int('conv2_out', 16, 32),
        'fc1_hidden': trial.suggest_int('fc1_hidden', 120, 240),
        'fc2_hidden': trial.suggest_int('fc2_hidden', 84, 100),
        'batch_size': trial.suggest_categorical('batch_size', [4, 16]),
        'epochs': trial.suggest_categorical('epochs', [2]),
        'lr': trial.suggest_float('lr', 0.001, 0.005),
        'momentum': trial.suggest_categorical('momentum', [0.9, 0.95, 0.99]),
        'optimizer': trial.suggest_categorical('optimizer', ['sgd', 'adam']),
        'loss': 'cross_entropy',
        'validation_split': 0.1,
        'seed': 42,
    }
    result = run_experiment(config)
    return result['val_acc']

def main():
    search_configs = generate_search_configs(DEFAULT_CONFIG, SEARCH_SPACE)
    results = []

    start_time = time.time()
    print(f"Total trials: {len(search_configs)}")
    for config in search_configs: #遍历所有配置
        print(f"\nRunning hyperparameter search: {config['name']}")
        result = run_experiment(config)
        results.append(result)
        print(
            f"[{config['name']}] val_loss={result['val_loss']:.4f}, "
            f"val_acc={result['val_acc'] * 100:.2f}%"
        )

    print_comparison(results)

    best_result = find_best_result(results)
    best_config = merge_config(
        DEFAULT_CONFIG,
        {
            'model': best_result['model'],
            'conv1_out': best_result['conv1_out'],
            'conv2_out': best_result['conv2_out'],
            'fc1_hidden': best_result['fc1_hidden'],
            'fc2_hidden': best_result['fc2_hidden'],
            'batch_size': best_result['batch_size'],
            'epochs': best_result['epochs'],
            'lr': best_result['lr'],
            'optimizer': best_result['optimizer'],
        },
    )
    best_config['name'] = 'best-config'

    print('\nBest hyperparameters based on validation accuracy')
    print(best_config)

    final_result = train_best_config_on_full_data(best_config)
    print(
        f"\nFinal test result with best hyperparameters: "
        f"test_loss={final_result['test_loss']:.4f}, "
        f"test_acc={final_result['test_acc'] * 100:.2f}%"
    )
    print('for 运行时间:', time.time() - start_time)

    optuna_time = time.time()
    print('------optuna:----------')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)  # 试20组参数
    print('最优参数:', study.best_params)
    print('最优分数:', study.best_value)
    print('optuna运行时间:', time.time() - optuna_time)


if __name__ == '__main__':
    main()
