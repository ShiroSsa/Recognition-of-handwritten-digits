# 加载数据集
from torch.utils.data import DataLoader
from data_reader import Digits_Dataset
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
train_dir = 'data/mnist_train.csv'
test_dir = 'data/mnist_test.csv'
train_data = Digits_Dataset(train_dir)
test_data = Digits_Dataset(test_dir)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)

# 加载数据集,每批次64个图像,每次读取随机洗牌
train_dataloader = DataLoader(train_data, 64, shuffle=True)
test_dataloader = DataLoader(test_data, 64, shuffle=True)

# 检查dataloader是否工作正常
if __name__ == '__main__':
    writer = SummaryWriter('logs')
    step = 0
    print(f'训练数据集长度:{train_data_size}')
    print(f'测试数据集长度:{test_data_size}')
    for digit_name, digit_image in train_dataloader:
        # print(digit_name, digit_image)
        # 加入一个通道维度,因为灰度图片只有一个通道
        digit_image = digit_image.unsqueeze(1)
        writer.add_images('image', digit_image, step)
        step += 1
    writer.close()
