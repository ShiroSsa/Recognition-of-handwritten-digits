# 从csv中读取数据
import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset


# 构造数据集
class Digits_Dataset(Dataset):

    # 数据内容：数据文件地址
    def __init__(self, file_dir):
        self.file_dir = file_dir
        # 打开csv
        self.df = pd.read_csv(file_dir, header=None)
        # 按行读取csv数据
        self.data = [row for index, row in self.df.iterrows()]

    # 获得数据集长度
    def __len__(self):
        return self.df.shape[0]

    # 获得标签（digit_name）和图像矩阵（digit_image）
    def __getitem__(self, idx):
        # csv每行的第一个数为标签
        digit_name = self.data[idx][0]
        # csv每行第二个数至最后一个数为手写体灰度，转换为28X28矩阵
        digit_image = np.array(self.data[idx][1:].tolist()).reshape((28, 28))
        # 转换为torch使用的数据类型tensor
        digit_image = torch.tensor(digit_image, dtype=torch.float32)
        return digit_name, digit_image


# 看看数据长什么样
if __name__ == '__main__':
    writer = SummaryWriter('logs')
    df = pd.read_csv('data/mnist_test.csv', header=None)
    # for index, row in df.iterrows():
    #     print(row)
    print('----------------------')
    print(f'数据集大小:{df.shape}')
    print('----------------------')
    testset = Digits_Dataset('data/mnist_test.csv')
    for digit_name, digit_image in testset:
        print(digit_name, digit_image)
        # 加入批次维度和通道维度,以能够写入SummaryWriter中
        digit_image = digit_image.unsqueeze(0).unsqueeze(0)
        writer.add_images('image', digit_image, digit_name)
    writer.close()
