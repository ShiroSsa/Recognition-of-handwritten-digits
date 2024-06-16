# 神经网络架构
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class Recognition(nn.Module):
    def __init__(self):
        super(Recognition, self).__init__()
        # 卷积层1:输入通道数1,输出通道数32,卷积核5X5,填充2不改变图像大小
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        # 最大池化层1:卷积核2X2,图像大小14X14
        self.maxpool1 = nn.MaxPool2d(2)
        # 卷积层2:输入通道数32,输出通道数32
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        # 最大池化层2:图像大小7X7
        self.maxpool2 = nn.MaxPool2d(2)
        # 卷积层3:输入通道数32,输出通道数64
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        # 最大池化层3:图像大小3X3
        self.maxpool3 = nn.MaxPool2d(2)
        # 拉平张量
        self.flatten = nn.Flatten()
        # 全连接层1:输入通道=3X3X64=576
        self.linear1 = nn.Linear(576, 64)
        # 全连接层2:输入通道64,输出通道10(因为识别0~9十个数)
        self.linear2 = nn.Linear(64, 10)

    # 前馈网络
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = torch.softmax(x, dim=1)
        return x


if __name__ == '__main__':
    writer = SummaryWriter('logs')
    test_module = Recognition()
    print(test_module)
    # 用28X28的全1矩阵调试
    input = torch.ones(64, 1, 28, 28)
    output = test_module(input)
    print(output)
    # 写入网络架构
    writer.add_graph(test_module, input)
    writer.close()
