#采用深度卷积神经网络（VGG）在 Fashion-MNIST数据集上实施训练，并对其 epoch与收敛性间的关系进行分析。
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
print("第1步：导入库包成功！\n-------------------------------------------------------------------------------------")

# 处理训练集数据
def train_data_process():
    # 加载FashionMNIST数据集
    train_data = FashionMNIST(root="./data",  # 数据路径
                              train=True,  # 只使用训练数据集
                              transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                              # 把PIL.Image或者numpy.array数据类型转变为torch.FloatTensor类型
                              # 尺寸为Channel * Height * Width，数值范围缩小为[0.0, 1.0]
                              # 这里将尺寸调整为224，是因为VGG网络的输入尺寸要求是224.
                              download=True,  # 若本身没有下载相应的数据集，则选择True
                              )
    train_loader = Data.DataLoader(dataset=train_data,  # 传入的数据集
                                   batch_size=64,  # 每个Batch中含有的样本数量
                                   shuffle=False,  # 不对数据集重新排序
                                   num_workers=0,  # 加载数据所开启的进程数量
                                   )
    print("The number of batch in train_loader:", len(train_loader))  # 一共有938个batch，每个batch含有64个训练样本

    # 获得一个Batch的数据
    for step, (b_x, b_y) in enumerate(train_loader):  #enumerate返回含有两个元素的元组，元组的第一项是索引，第二项是（当前批次的输入数据，对应标签）
        if step > 0:
            break
    batch_x = b_x.squeeze().numpy()  # 将四维张量移除第1维。如将（批次大小 x 通道数 x 高度 x 宽度）转换为三维张量（批次大小 x 高度 x 宽度）
    #在处理图像数据时，通常不需要保留通道维度，可以直接将图像展平为二维矩阵。
    batch_y = b_y.numpy()  # 将张量转换成Numpy数组
    class_label = train_data.classes  # 训练集的标签
    class_label[0] = "T-shirt"
    print("the size of batch in train data:", batch_x.shape)

    # 可视化一个Batch的图像
    plt.figure(figsize=(12, 5))  #创建一个图形窗口，设置图形的大小为12x5英寸。
    for ii in np.arange(len(batch_y)):  #遍历batch_y的长度，batch_y是一个包含类别标签的数组。
        plt.subplot(4, 16, ii + 1)  #创建一个4行16列的子图网格，并在第ii+1个子图中绘制图像。
        plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)  #在当前子图中显示batch_x的第ii个图像，使用灰度颜色映射。
        plt.title(class_label[batch_y[ii]], size=9)  #为当前子图设置标题，标题为class_label数组中对应batch_y[ii]的类别标签，字体大小为9。
        plt.axis("off")  #关闭当前子图的坐标轴。
        plt.subplots_adjust(wspace=0.05)  #调整子图之间的水平间距为0.05英寸。
    plt.show()

    return train_loader, class_label
print("第2步：训练数据集准备成功！\n-------------------------------------------------------------------------------------")

# 处理测试集数据
def test_data_process():
    test_data = FashionMNIST(root="./data",  # 数据路径
                             train=False,  # 不使用训练数据集
                             transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),  # 把PIL.Image或者numpy.array数据类型转变为torch.FloatTensor类型                                                                                   # 尺寸为Channel * Height * Width，数值范围缩小为[0.0, 1.0]
                             download=True,  # 如果前面数据已经下载，这里不再需要重复下载
                             )
    test_loader = Data.DataLoader(dataset=test_data,  # 传入的数据集
                                  batch_size=1,  # 每个Batch中含有的样本数量
                                  shuffle=True,  # 对数据集重新排序
                                  num_workers=0,  # 加载数据所开启的进程数量
                                   )
    # 获得一个Batch的数据
    for step, (b_x, b_y) in enumerate(test_loader):
        if step > 0:
            break
    batch_x = b_x.squeeze().numpy()  # 将四维张量移除第1维，并转换成Numpy数组
    batch_y = b_y.numpy()  # 将张量转换成Numpy数组
    print("The size of batch in test data:", batch_x.shape)
    return test_loader
print("第3步：测试集准备成功！\n-------------------------------------------------------------------------------------")

# 定义一个VGG模块
def vgg_block(num_convs, in_channels, out_channels):
    '''
    :param num_convs: 卷积层的数量
    :param in_channels: 输入图像的通道数
    :param out_channels: 输出图像的通道数
    '''

    blk = []
    for num in range(num_convs):
        if num == 0:  # 定义第一个卷积块的输入输出通道数
            '''如果是第一个卷积层，则添加一个输入通道数为in_channels，输出通道数为out_channels的二维卷积层，卷积核大小为3x3，边缘填充为1。'''
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:  # 定义后面几个卷积块的输入输出通道数
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 定义一个最大池化层

    return nn.Sequential(*blk)
    '''使用星号操作符将列表blk中的元素解包，即将列表中的每个元素作为独立的参数传递给nn.Sequential()函数。
        nn.Sequential()函数用于将多个神经网络层按顺序组合成一个模块。传入的参数是各个层的对象。'''
print("第4步：VGG模块创建成功！\n-------------------------------------------------------------------------------------")

# 定义VGG-11网络结构
class VGG_Net(nn.Module):
    def __init__(self, conv_arch, fc_features, fc_hidden_units=4096):
        '''定义了类的初始化方法，接收三个参数：conv_arch（卷积层配置），fc_features（全连接层输入特征数）和
        fc_hidden_units（全连接层隐藏单元数，默认为4096）。'''
        super(VGG_Net, self).__init__()
        self.conv = nn.Sequential()  # 从名称 conv看，定义了卷积层，用于存储卷积层。
        self.fc = nn.Sequential()    # 从名称 fc看，定义了全连接层，用于存储全连接层。
        # 卷积层部分
        for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch): # 遍历conv_arch列表，获取每个VGG模块的配置信息。
            # 根据配置增加VGG模块
            self.conv.add_module("vgg_block_" + str(i + 1), vgg_block(num_convs, in_channels, out_channels))
        # 全连接层部分
        self.fc = nn.Sequential(nn.Flatten(),  # 首先添加一个Flatten层，将输入展平。
                                nn.Linear(fc_features, fc_hidden_units),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(fc_hidden_units, fc_hidden_units),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(fc_hidden_units, 10)
                                )
    # 前向传播路径
    def forward(self, x):
        x = self.conv(x)
        output = self.fc(x.view(x.size(0), -1))  # 将卷积层的输出x展平，传入全连接层，得到输出output。
        return output
print("第5步：VGG-11网络搭建成功！\n-------------------------------------------------------------------------------------")

# 定义网络的训练过程
def train_model(model, traindataloader, train_rate, criterion, device, optimizer, num_epochs=25):
    '''
    :param model: 网络模型
    :param traindataloader: 训练数据集，会切分为训练集和验证集
    :param train_rate: 训练集batch_size的百分比
    :param criterion: 损失函数
    :param device: 运行设备
    :param optimizer: 优化方法
    :param num_epochs: 训练的轮数
    '''

    batch_num = len(traindataloader)  # batch数量
    train_batch_num = round(batch_num * train_rate)  # 将80%的batch用于训练，round()函数四舍五入
    best_model_wts = copy.deepcopy(model.state_dict())  # 复制当前模型的参数
    # 初始化参数
    best_acc = 0.0  # 最高准确度
    train_loss_all = []  # 训练集损失函数列表
    train_acc_all = []  # 训练集准确度列表
    val_loss_all = []  # 验证集损失函数列表
    val_acc_all = []  # 验证集准确度列表
    since = time.time()  # 当前时间
    # 进行迭代训练模型
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 初始化参数
        train_loss = 0.0  # 训练集损失函数
        train_corrects = 0  # 训练集准确度
        train_num = 0  # 训练集样本数量
        val_loss = 0.0  # 验证集损失函数
        val_corrects = 0  # 验证集准确度
        val_num = 0  # 验证集样本数量
        # 对每一个mini-batch训练和计算
        for step, (b_x, b_y) in enumerate(traindataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            if step < train_batch_num:  # 使用数据集的80%用于训练
                model.train()  # 设置模型为训练模式，启用Batch Normalization和Dropout
                output = model(b_x)  # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
                pre_lab = torch.argmax(output, 1)  # 查找每一行中最大值对应的行标
                loss = criterion(output, b_y)  # 计算每一个batch的损失函数
                optimizer.zero_grad()  # 将梯度初始化为0
                loss.backward()  # 反向传播计算
                optimizer.step()  # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
                train_loss += loss.item() * b_x.size(0)  # 对损失函数进行累加
                train_corrects += torch.sum(pre_lab == b_y.data)  # 如果预测正确，则准确度train_corrects加1
                train_num += b_x.size(0)  # 当前用于训练的样本数量
            else:  # 使用数据集的20%用于验证
                model.eval()  # 设置模型为评估模式，不启用Batch Normalization和Dropout
                output = model(b_x)  # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
                pre_lab = torch.argmax(output, 1)  # 查找每一行中最大值对应的行标
                loss = criterion(output, b_y)  # 计算每一个batch中64个样本的平均损失函数
                val_loss += loss.item() * b_x.size(0)  # 将验证集中每一个batch的损失函数进行累加
                val_corrects += torch.sum(pre_lab == b_y.data)  # 如果预测正确，则准确度val_corrects加1
                val_num += b_x.size(0)  # 当前用于验证的样本数量

        # 计算并保存每一次迭代的成本函数和准确率
        train_loss_all.append(train_loss / train_num)  # 计算并保存训练集的成本函数
        train_acc_all.append(train_corrects.double().item() / train_num)  # 计算并保存训练集的准确率
        val_loss_all.append(val_loss / val_num)  # 计算并保存验证集的成本函数
        val_acc_all.append(val_corrects.double().item() / val_num)  # 计算并保存验证集的准确率
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 寻找最高准确度
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]  # 保存当前的最高准确度
            best_model_wts = copy.deepcopy(model.state_dict())  # 保存当前最高准确度下的模型参数
        time_use = time.time() - since  # 计算耗费时间
        print("Train and val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    # 选择最优参数
    model.load_state_dict(best_model_wts)  # 加载最高准确度下的模型参数
    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all}
                                 )  # 将每一代的损失函数和准确度保存为DataFrame格式

    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

    return model, train_process

# 测试模型
def test_model(model, testdataloader, device):
    '''
    :param model: 网络模型
    :param testdataloader: 测试数据集
    :param device: 运行设备
    '''

	# 初始化参数
    test_corrects = 0.0
    test_num = 0
    test_acc = 0.0
    # 只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in testdataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()  # 设置模型为评估模式，不启用Batch Normalization和Dropout
            output = model(test_data_x)  # 前向传播过程，输入为测试数据集，输出为对每个样本的预测
            pre_lab = torch.argmax(output, 1)  # 查找每一行中最大值对应的行标
            test_corrects += torch.sum(pre_lab == test_data_y.data)  # 如果预测正确，则准确度val_corrects加1
            test_num += test_data_x.size(0)  # 当前用于训练的样本数量

    test_acc = test_corrects.double().item() / test_num  # 计算在测试集上的分类准确率
    print("test accuracy:", test_acc)

# 模型的训练和测试
def train_model_process(myconvnet):
    optimizer = torch.optim.Adam(myconvnet.parameters(), lr=0.001)  # 使用Adam优化器，学习率为0.001
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵函数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # GPU加速
    train_loader, class_label = train_data_process()  # 加载训练集
    test_loader = test_data_process()  # 加载测试集

    myconvnet = myconvnet.to(device)
    myconvnet, train_process = train_model(myconvnet, train_loader, 0.8, criterion, device, optimizer, num_epochs=25)  # 开始训练模型
    test_model(myconvnet, test_loader, device)  # 使用测试集进行评估
print("第6步：定义训练过程和预测过程成功！\n-------------------------------------------------------------------------------------")

if __name__ == '__main__':
    ratio = 8
    small_conv_arch = [(1, 1, 64 // ratio), (1, 64 // ratio, 128 // ratio), (2, 128 // ratio, 256 // ratio),
                       (2, 256 // ratio, 512 // ratio), (2, 512 // ratio, 512 // ratio)]
    fc_features = 512 * 7 * 7
    fc_hidden_units = 4096

    vggnet = VGG_Net(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
    train_model_process(vggnet)
print("第7步：主函数运行成功！\n-------------------------------------------------------------------------------------")