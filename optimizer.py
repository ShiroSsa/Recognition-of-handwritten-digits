# 优化器
import torch.optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import data_loader
import NN_module

writer = SummaryWriter('logs')
# 超参数设置
learning_rate = 0.001
epoch_range = 20

# 实例化神经网络
recognition = NN_module.Recognition()
if torch.cuda.is_available():
    recognition = recognition.cuda()

# 表现指数(损失)使用交叉熵计算,优化方法采用随机梯度下降算法
loss = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss = loss.cuda()
optim = torch.optim.SGD(recognition.parameters(), learning_rate)

# 机器学习
# 训练次数
total_train_step = 0
# 测试次数
total_val_step = 0
for epoch in range(epoch_range):
    print(f'----------------第{epoch + 1}轮训练开始----------------')
    # 每轮学习总损失
    loss_per_turn = 0.0
    # 学习开始
    # 导入一批训练数据
    for data in data_loader.train_dataloader:
        # 导入目标(target)和图像
        labels, images = data
        # 加入通道维度
        images = images.unsqueeze(1)
        if torch.cuda.is_available():
            labels = labels.cuda()
            images = images.cuda()
        # 计算输出
        output = recognition(images)
        # 计算表现指数
        res_loss = loss(output, labels)
        # 优化器梯度清零
        optim.zero_grad()
        # 反向传播计算梯度
        res_loss.backward()
        # 优化参数
        optim.step()
        loss_per_turn += res_loss
        # 记录学习次数
        total_train_step += 1
    # 记录表现指数
    print(f'学习次数{total_train_step},Loss:{loss_per_turn:.10f}')
    writer.add_scalar('train_loss', loss_per_turn, total_train_step)

    # 测试开始
    total_accuracy = 0.0
    with torch.no_grad():
        for data in data_loader.test_dataloader:
            # 导入目标(target)和图像,与训练集一样
            labels, images = data
            images = images.unsqueeze(1)
            if torch.cuda.is_available():
                labels = labels.cuda()
                images = images.cuda()
            output = recognition(images)
            # 找到所有正确的预测并加和
            accuracy = (output.argmax(1) == labels).sum()
            total_accuracy += accuracy
    total_val_step += 1
    print(f'测试集正确率:{100 * total_accuracy / data_loader.test_data_size:.2f}%')
    writer.add_scalar('正确率', total_accuracy / data_loader.test_data_size, total_val_step)

    # 保存每次训练的模型
    torch.save(recognition, f'pth/test_{format(epoch)}.pth')
    print('模型已保存')


writer.close()
